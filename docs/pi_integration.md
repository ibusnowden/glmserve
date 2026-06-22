# Pi Code / coding-agent integration

glmserve exposes an OpenAI-compatible endpoint, so any harness that speaks the
OpenAI Chat Completions API (Pi Code, Cline, OpenCode, LiteLLM, the OpenAI SDK)
can drive it.

```
Pi Code / Cline / OpenCode  ──►  OpenAI-compatible API  ──►  glmserve  ──►  GPU cluster
```

## 1. Start the server

```bash
source scripts/env.sh
./build/glmserve serve --model /path/to/GLM-5.2-w4a16 \
  --port 8000 --ctx 131072 --max-seqs 1 --name glm-5.2-local
```

## 2. Point the OpenAI SDK at it

```python
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

r = client.chat.completions.create(
    model="glm-5.2-local",
    messages=[{"role": "user", "content": "Inspect this repo and propose a minimal patch."}],
    temperature=0.2, max_tokens=4096, stream=True,
)
for chunk in r:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## 3. Pi-style model config

```json
{
  "models": {
    "glm-5.2-local": {
      "provider": "openai-compatible",
      "baseURL": "http://localhost:8000/v1",
      "apiKey": "EMPTY",
      "model": "glm-5.2-local"
    }
  }
}
```

The exact config path differs per harness/version, but the concept is identical:
an OpenAI-compatible base URL + a model id. For remote nodes, front glmserve with
LiteLLM or an SSH tunnel; the streaming format is standard SSE so no adapter is
needed.

## Notes for coding agents

- `temperature: 0` (greedy) gives reproducible patches; raise for brainstorming.
- Use `stop` sequences to cut generation at a fence/marker; glmserve trims them
  and never streams a partial stop string.
- Keep `max-seqs` low (1–4): coding-agent traffic is latency-sensitive and bursty.
- Long-context repo reads: raise `--ctx` (KV cache scales with it); watch VRAM.
- Tool-calling / reasoning-mode parsers (`glm47`/`glm45` style) are Phase 7;
  until then, prompt the model to emit tool calls as plain JSON and parse harness-side.
