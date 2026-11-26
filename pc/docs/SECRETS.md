# Tokens, API keys and provider settings

Lenswright keeps local credentials in **`pc/settings.local.json`**. The file is
git-ignored and is never returned unmasked by the API.

## Where to set it

| Platform | Location |
|---|---|
| PC WebUI | **Settings** tab → "HuggingFace token" and "LLM provider" |
| PC CLI | create `pc/settings.local.json` by hand (same schema) |
| Android | Settings card → **API & model settings** → HF token / provider / base URL / API key |

The PC server applies the stored values to the environment on startup, so the
existing consumers (`download_models.py`, OCR/VLM loaders, `llm_providers.py`)
pick them up without any other change.

## Schema

```json
{
  "hf_token": "hf_...",
  "llm": {
    "provider": "lmstudio",
    "base_url": "http://127.0.0.1:1234/v1",
    "model": "qwen3.5-9b",
    "api_key": "",
    "endpoint": ""
  }
}
```

Environment variables written at startup:

| Field | Env var |
|---|---|
| `hf_token` | `HF_TOKEN` (and `HUGGING_FACE_HUB_TOKEN`) |
| `llm.provider` | `LENSWRIGHT_LLM_PROVIDER` |
| `llm.base_url` | `LENSWRIGHT_LLM_BASE_URL` |
| `llm.model` | `LENSWRIGHT_LLM_MODEL` |
| `llm.api_key` | `LENSWRIGHT_LLM_API_KEY` |
| `llm.endpoint` | `LENSWRIGHT_LLM_ENDPOINT` |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/settings/secrets` | Masked view (`****abcd`) + `_configured` flags |
| PUT | `/v1/settings/secrets` | Save; masked values are ignored, empty strings clear |
| POST | `/v1/settings/secrets/test/hf` | Validate the token with `whoami` |
| POST | `/v1/settings/secrets/test/llm` | Send a one-word prompt to the provider |

## Which models need the token

| Model | Gated? | Note |
|---|---|---|
| `flux-kontext-dev` (full bf16) | yes | `black-forest-labs/FLUX.1-Kontext-dev` |
| `flux-kontext-dev-fp8` | no | ungated mirror, same pipeline |
| `cohere-transcribe` | yes | CohereLabs repo |
| everything else | no | public repos |

## Security notes

- The file is created with mode `0600` on POSIX; Windows uses the user profile ACL.
- `GET` never returns the raw secret; the UI shows `saved: ****abcd`.
- Sending the masked placeholder back does **not** overwrite the stored value,
  so a browser refresh cannot wipe a token.
- Nothing is uploaded anywhere; the token is only sent to the provider you
  configured (HuggingFace, or the LLM base URL).
- Do not commit `settings.local.json`. It is already listed in `.gitignore`.
