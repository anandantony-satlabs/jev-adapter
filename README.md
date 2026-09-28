# pi-jev-adapter

pi extension port of [`dsh-jev-adapter`](https://github.com/BetterZflyee/dsh-jev-adapter) (MIT).
Registers one tool, `jev_decide`, that runs the **Jev (System One) decision-model
paradigm** over your local OpenAI-compatible endpoint instead of the TypeSafe Jev API.
Repeatable sandbox fixtures live in `tests/jev-adapter/` (run via extension_sandbox with
`fixturesDir`).

## How the endpoint is resolved

1. `JEV_BASE_URL` / `JEV_MODEL` / `JEV_API_KEY` env vars, if set
2. otherwise the provider in `~/.pi/agent/models.json` whose model id contains `GLM-5.3-Flash`
3. otherwise the tool fails with a clear setup error — **no endpoint is hard-coded**

Optional tuning: `JEV_MAX_TOKENS` (4000), `JEV_TIMEOUT_MS` (120000), `JEV_RETRIES` (3),
`JEV_MODELS_JSON` (alternate models.json path), `JEV_REASONING_EFFORT` (default `low`),
`JEV_RESPONSE_FORMAT` (default `json_object`; `none` to disable).

`reasoning_effort` is the latency control: hidden reasoning dominates wall-clock time on
GLM-5.3 servers (measured: default ~36s / ~1100 completion tokens vs `low` ~1.2s / 41
tokens for the same question). Accepted values: `none | minimal | low | medium | high |
xhigh | max`. CAVEAT: `none` does NOT disable thinking — it zeroes the reasoning-token
accounting but the thinking leaks into `content` as prose (unparseable); use `low` for
speed, `medium`/`high` for nuanced judgements.

`response_format: json_object` (default) asks the vLLM server for syntactically-valid
JSON — removes the parse-retry tier at zero latency/behavior cost (probe: honest spreads
preserved, same token count). Do NOT set `json_schema`: constrained decoding at temp 0
collapses every answer to one-hot argmax and cannot express sum-to-1.

Each result reports `elapsed`, `attempts` (requests actually made — a value >1 means the
parse-retry rescued the call), and token `usage` in its `perf:` line. Run `/jev-config`
to see the resolved values.

## Usage

Ask the agent to make atomic judgements; it batches them into one `jev_decide` call:

```
Classify these 12 support tickets: department + urgency + customer frustration.
```

**Batch mode (multiple states):** pass `states: [{id, state}, ...]` to judge many
independent states (commits, tickets, diffs) against the SAME questions in one call —
answers come back keyed by state id. One batched call beats N parallel tool calls (the
local endpoint serializes on the GPU, so parallel calls just queue). Output decode still
scales with states×questions, so keep it lean: ≤10 states × ≤6 questions per call.

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
