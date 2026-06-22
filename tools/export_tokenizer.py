#!/usr/bin/env python3
"""Tokenizer bridge for glmserve (spec §9.2 V0 path).

glmserve has a built-in byte-level fallback and a best-effort tokenizer.json BPE
loader. For *exact* GLM-5.2 tokenization before the C++ tokenizer is fully wired,
this script uses the HF tokenizer to:
  * encode a prompt (or chat messages) to token ids you can pass to
    `glmserve dump --tokens "..."`, and
  * report special token ids (bos/eos) for config sanity-checking.

Usage:
  python tools/export_tokenizer.py --model DIR --text "fix the failing test"
  python tools/export_tokenizer.py --model DIR --chat '[{"role":"user","content":"hi"}]'
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", default=None)
    ap.add_argument("--chat", default=None, help="JSON list of {role,content}")
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
    except Exception as e:  # noqa
        print(f"transformers unavailable ({e}); install it for exact tokenization.")
        return 1

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"tokenizer: vocab_size={tok.vocab_size} "
          f"bos={tok.bos_token_id} eos={tok.eos_token_id}")

    if args.chat:
        msgs = json.loads(args.chat)
        try:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        except Exception:
            text = "".join(f"<|{m['role']}|>\n{m['content']}" for m in msgs) + "<|assistant|>\n"
            ids = tok.encode(text)
    else:
        ids = tok.encode(args.text if args.text is not None else "Hello!")

    print("ids:", " ".join(str(i) for i in ids))
    print(f"count: {len(ids)}")
    print('feed to engine:  glmserve dump --model %s --tokens "%s"'
          % (args.model, " ".join(str(i) for i in ids)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
