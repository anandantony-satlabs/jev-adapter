# pi-jev-adapter

pi extension port of [`dsh-jev-adapter`](https://github.com/BetterZflyee/dsh-jev-adapter) (MIT).
Registers one tool, `jev_decide`, that runs the **Jev (System One) decision-model
paradigm** over your local OpenAI-compatible endpoint instead of the TypeSafe Jev API.

## How the endpoint is resolved

1. `JEV_BASE_URL` / `JEV_MODEL` / `JEV_API_KEY` env vars, if set
2. otherwise the provider in `~/.pi/agent/models.json` whose model id contains `GLM-5.3-Flash`
3. otherwise the tool fails with a clear setup error — **no endpoint is hard-coded**

Optional tuning: `JEV_MAX_TOKENS` (4000), `JEV_TIMEOUT_MS` (120000), `JEV_RETRIES` (3),
`JEV_MODELS_JSON` (alternate models.json path). Run `/jev-config` to see the resolved values.

## Usage

Ask the agent to make atomic judgements; it batches them into one `jev_decide` call:

```
Classify these 12 support tickets: department + urgency + customer frustration.
```

| Question type | You provide | You get back |
|---|---|---|
| `choice` | options map (≤255) | picked option + probability distribution + confidence |
| `score` | ordered levels (2–10) | fractional score + per-level probabilities + confidence |
| `boolean` | a statement | P(true) |

Route on confidence: `≥0.85 → act`, `0.5–0.85 → draft + human confirm`, `<0.5 → escalate`.
Tune thresholds to the risk of the action.

## Security notes

- **No secrets in this repo.** The endpoint URL comes from the user's own
  `~/.pi/agent/models.json`; API keys are read from there or from env vars at runtime
  and are never logged, rendered, or sent to the primary model (`/jev-config` masks them).
- **The resolved `apiKey` is sent as a Bearer header to whatever `baseURL` resolves to.**
  Only point `JEV_BASE_URL` at endpoints you trust — a hostile endpoint receives both
  the key and the judged state payload.
- **Prompt injection is inherent to the paradigm.** `state`, `instructions`, and
  `criteria` are model-controlled and interpolated into the decision prompt; a
  compromised primary agent could craft input that coaxes high confidence out of the
  decision model. Probabilities are **self-reported estimates, not calibrated** —
  every result carries this caveat. Keep a human in the loop for high-risk actions.
- The extension performs network calls to the configured decision endpoint only.
  No `eval`, no child processes, no filesystem writes; the only file read is
  `models.json` (contents are never echoed).

## Credits

Ported from `dsh-jev-adapter` by BetterZflyee (MIT) — honesty prompt, reasoning-model
tolerance, probability normalisation, and retry behaviour are all from the original.
