#!/usr/bin/env python3
"""Fetch missing safetensors shards named by model.safetensors.index.json.

The glmserve real-weight flow often starts with config/tokenizer/index files on
disk but without the large shard files. This helper makes that state explicit and
recoverable: it reads the local index, lists missing shard filenames, and, when
given a repository or URL template, downloads only the missing files.

Examples:
  python tools/fetch_safetensors_shards.py --model /data/GLM-5.2 --dry-run
  python tools/fetch_safetensors_shards.py --model /data/GLM-5.2 --repo zai-org/GLM-5.2
  HF_TOKEN=... python tools/fetch_safetensors_shards.py --model /data/GLM-5.2 --repo zai-org/GLM-5.2
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_REPO = "zai-org/GLM-5.2"
DEFAULT_REVISION = "main"
CHUNK = 16 * 1024 * 1024


def shard_files(model_dir):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise SystemExit(f"missing index: {index_path}")
    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    weight_map = idx.get("weight_map")
    if not isinstance(weight_map, dict):
        raise SystemExit(f"index missing weight_map: {index_path}")
    return sorted(set(weight_map.values()))


def indexed_total_size(model_dir):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    total = idx.get("metadata", {}).get("total_size")
    return int(total) if total is not None else None


def token_from_env():
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(name)
        if v:
            return v
    return None


def build_url(filename, repo, revision, base_url):
    quoted_file = urllib.parse.quote(filename)
    if base_url:
        if "{" in base_url:
            return base_url.format(
                filename=quoted_file,
                raw_filename=filename,
                repo=urllib.parse.quote(repo or "", safe="/"),
                revision=urllib.parse.quote(revision, safe=""),
            )
        return base_url.rstrip("/") + "/" + quoted_file
    repo = repo or DEFAULT_REPO
    revision = revision or DEFAULT_REVISION
    return "https://huggingface.co/{}/resolve/{}/{}".format(
        urllib.parse.quote(repo, safe="/"),
        urllib.parse.quote(revision, safe=""),
        quoted_file,
    )


def request_with_headers(url, token, resume_from):
    headers = {"User-Agent": "glmserve-fetch-shards/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"
    return urllib.request.Request(url, headers=headers)


def probe_one(url, token, timeout):
    headers = {"User-Agent": "glmserve-fetch-shards/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        length = resp.headers.get("Content-Length")
        return getattr(resp, "status", 200), int(length) if length else None


def response_total_size(resp, resume_from):
    content_range = resp.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[1].strip()
        if total.isdigit():
            return int(total)
    length = resp.headers.get("Content-Length")
    if length and length.isdigit():
        return resume_from + int(length)
    return None


def download_one(url, dst, token, retries, timeout, overwrite, expected_size=None):
    part = dst + ".part"
    if overwrite:
        for p in (dst, part):
            if os.path.exists(p):
                os.remove(p)
    if os.path.exists(dst):
        if expected_size is not None and os.path.getsize(dst) != expected_size:
            bad = dst + ".bad-size"
            if os.path.exists(bad):
                os.remove(bad)
            os.replace(dst, bad)
        else:
            size = os.path.getsize(dst)
            return f"present {size / 1024**3:.2f} GiB"

    if os.path.exists(dst):
        return "present"

    for attempt in range(1, retries + 1):
        resume_from = os.path.getsize(part) if os.path.exists(part) else 0
        req = request_with_headers(url, token, resume_from)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                mode = "ab" if resume_from > 0 and status == 206 else "wb"
                if resume_from > 0 and status != 206:
                    resume_from = 0
                want_size = expected_size or response_total_size(resp, resume_from)
                downloaded = resume_from
                with open(part, mode) as f:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                if want_size is not None and downloaded != want_size:
                    raise IOError(
                        f"short download: got {downloaded} bytes, expected {want_size}"
                    )
                os.replace(part, dst)
                return f"downloaded {downloaded / 1024**3:.2f} GiB"
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise SystemExit(
                    f"download denied for {url} ({e.code}); set HF_TOKEN if the repo requires auth"
                )
            last = f"HTTP {e.code}: {e.reason}"
        except Exception as e:
            last = str(e)
        if attempt < retries:
            wait = min(60, 2 ** (attempt - 1))
            print(f"  retry {attempt}/{retries} after {last}; sleeping {wait}s", flush=True)
            time.sleep(wait)
    raise SystemExit(f"failed to download {url}: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint directory containing model.safetensors.index.json")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face repo id")
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--base-url", default=None,
                    help="optional URL prefix/template; supports {filename}, {raw_filename}, {repo}, {revision}")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe", type=int, default=0,
                    help="HEAD-check the first N selected missing shards and exit without downloading")
    ap.add_argument("--max-files", type=int, default=0,
                    help="download at most N missing files; useful for connectivity checks")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel downloads; keep modest on shared filesystems")
    ap.add_argument("--verify-present", action="store_true",
                    help="HEAD-check existing selected shards and refetch size mismatches")
    ap.add_argument("--first-shard", type=int, default=1,
                    help="1-based first shard ordinal from the index to consider")
    ap.add_argument("--last-shard", type=int, default=0,
                    help="1-based last shard ordinal from the index to consider; 0 means through the end")
    ap.add_argument("--require-free-gib", type=float, default=0.0,
                    help="fail before downloading unless the model filesystem has at least this much free space")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    files = shard_files(args.model)
    if args.first_shard < 1:
        raise SystemExit("--first-shard is 1-based and must be >= 1")
    last = args.last_shard if args.last_shard > 0 else len(files)
    if last < args.first_shard:
        raise SystemExit("--last-shard must be >= --first-shard")
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    selected = files[args.first_shard - 1:last]
    missing = [f for f in selected if not os.path.exists(os.path.join(args.model, f))]
    partials = [f + ".part" for f in files if os.path.exists(os.path.join(args.model, f + ".part"))]
    partial_bytes = sum(os.path.getsize(os.path.join(args.model, p)) for p in partials)
    total_size = indexed_total_size(args.model)
    os.makedirs(args.model, exist_ok=True)
    usage = shutil.disk_usage(args.model)
    print(f"model: {args.model}")
    print(f"  shard files present: {len(files) - len([f for f in files if not os.path.exists(os.path.join(args.model, f))])}/{len(files)}")
    if selected != files:
        print(f"  selected shard ordinals: {args.first_shard}..{last} ({len(selected)} file(s))")
    if total_size is not None:
        print(f"  indexed tensor bytes: {total_size / 1024**3:.2f} GiB")
    print(f"  filesystem free: {usage.free / 1024**3:.2f} GiB")
    if partials:
        print(f"  partial downloads: {len(partials)} file(s), {partial_bytes / 1024**3:.2f} GiB")
    if missing:
        print(f"  missing in selected range: {len(missing)}")
        for f in missing[:10]:
            print(f"    - {f}")
    else:
        print("  all selected shard files are present")
        if not args.verify_present:
            return 0

    if args.require_free_gib > 0 and usage.free < args.require_free_gib * 1024**3:
        raise SystemExit(
            f"insufficient free space: need at least {args.require_free_gib:.2f} GiB, "
            f"found {usage.free / 1024**3:.2f} GiB"
        )

    if args.dry_run:
        print("dry-run: no downloads attempted")
        return 0

    to_fetch = missing[:args.max_files] if args.max_files > 0 else missing
    token = token_from_env()
    print(f"  source: {args.base_url or ('https://huggingface.co/' + args.repo + '/resolve/' + args.revision)}")
    expected_sizes = {}
    if args.verify_present:
        present_selected = [f for f in selected if os.path.exists(os.path.join(args.model, f))]
        print(f"  verifying existing selected shards: {len(present_selected)} file(s)")
        bad_present = []
        for filename in present_selected:
            url = build_url(filename, args.repo, args.revision, args.base_url)
            status, nbytes = probe_one(url, token, args.timeout)
            if nbytes is None:
                continue
            expected_sizes[filename] = nbytes
            dst = os.path.join(args.model, filename)
            local = os.path.getsize(dst)
            if local != nbytes:
                bad_present.append((filename, local, nbytes))
        if bad_present:
            print(f"  size mismatches: {len(bad_present)}")
            for filename, local, nbytes in bad_present[:10]:
                print(f"    - {filename}: local {local / 1024**3:.2f} GiB != remote {nbytes / 1024**3:.2f} GiB")
            if args.dry_run or args.probe > 0:
                print("verify-present: no files changed")
            else:
                for filename, _, nbytes in bad_present:
                    dst = os.path.join(args.model, filename)
                    bad = dst + ".bad-size"
                    if os.path.exists(bad):
                        os.remove(bad)
                    os.replace(dst, bad)
                    expected_sizes[filename] = nbytes
                # Respect --max-files for initially missing files, but always
                # repair explicitly detected corrupt present files.
                to_fetch = to_fetch + [filename for filename, _, _ in bad_present]
        else:
            print("  existing selected shards match remote sizes")
    if args.probe > 0:
        probe_files = to_fetch[:args.probe]
        print(f"  probing: {len(probe_files)} file(s)")
        for i, filename in enumerate(probe_files, 1):
            url = build_url(filename, args.repo, args.revision, args.base_url)
            try:
                status, nbytes = probe_one(url, token, args.timeout)
                if nbytes is not None:
                    expected_sizes[filename] = nbytes
                size = "unknown size" if nbytes is None else f"{nbytes / 1024**3:.2f} GiB"
                print(f"[{i}/{len(probe_files)}] {filename}: HTTP {status}, {size}")
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise SystemExit(
                        f"probe denied for {url} ({e.code}); set HF_TOKEN if the repo requires auth"
                    )
                raise
        print("probe: no downloads attempted")
        return 0

    if args.verify_present:
        for filename in to_fetch:
            if filename in expected_sizes:
                continue
            url = build_url(filename, args.repo, args.revision, args.base_url)
            _, nbytes = probe_one(url, token, args.timeout)
            if nbytes is not None:
                expected_sizes[filename] = nbytes

    print(f"  downloading: {len(to_fetch)} file(s), jobs={args.jobs}")

    def fetch_one(item):
        i, filename = item
        dst = os.path.join(args.model, filename)
        os.makedirs(os.path.dirname(dst) or args.model, exist_ok=True)
        url = build_url(filename, args.repo, args.revision, args.base_url)
        result = download_one(url, dst, token, args.retries, args.timeout, args.overwrite,
                              expected_size=expected_sizes.get(filename))
        return i, filename, result

    if args.jobs == 1:
        for item in enumerate(to_fetch, 1):
            i, filename = item
            print(f"[{i}/{len(to_fetch)}] {filename}", flush=True)
            i, filename, result = fetch_one(item)
            print(f"  {result}", flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = [pool.submit(fetch_one, item) for item in enumerate(to_fetch, 1)]
            for fut in concurrent.futures.as_completed(futs):
                i, filename, result = fut.result()
                print(f"[{i}/{len(to_fetch)}] {filename}: {result}", flush=True)

    remaining = [f for f in selected if not os.path.exists(os.path.join(args.model, f))]
    if remaining:
        print(f"remaining missing selected shards: {len(remaining)}")
        if args.max_files > 0:
            print("bounded fetch complete: requested file limit reached")
            return 0
        return 1
    if selected == files:
        print("all shard files are present")
    else:
        print("all selected shard files are present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
