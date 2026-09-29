# pi-jev-adapter

pi extension port of [`dsh-jev-adapter`](https://github.com/BetterZflyee/dsh-jev-adapter) (MIT).
Registers one tool, `jev_decide`, that runs the **Jev (System One) decision-model
paradigm** over your local OpenAI-compatible endpoint instead of the TypeSafe Jev API.
Which model does the deciding is **configuration, not code** — see
[`jev-adapter.config.json`](jev-adapter.config.json). Repeatable sandbox fixtures live in
`tests/jev-adapter/fixtures/` (run via `extension_sandbox` with `fixturesDir`), plus two
fetch-mocked unit tests that need no endpoint.

## Which model decides (no model names in the source)

`jev-adapter.config.json` next to `index.ts` is the registry:

```json
{
  "defaults": { "reasoningEffort": "low", "maxTokens": 4000, "timeoutMs": 180000, "retries": 3, "responseFormat": "json_object" },
  "decisionModels": [
    {
      "match": "Qwen3.8-Flash-Next-NVFP4",
      "reasoningEffort": "low",
      "supportedEfforts": ["none", "low", "medium", "xhigh"],
      "effortMap": { "minimal": "low", "high": "xhigh", "max": "xhigh" },
      "effortGuide": "short how-to-pick text, rendered into the tool description (prompt tokens — keep it lean)",
      "note": "long-form measured behaviour of this endpoint — shown by /jev-config"
    },
    { "match": "GLM-5.3-Flash", "reasoningEffort": "low", "note": "fallback / rollback entry" }
  ]
}
```

* `decisionModels` is a **preference list**: the first entry whose `match` (exact id, or a
  substring of one) appears in a model id in `~/.pi/agent/models.json` wins. **Switching
  endpoints = reorder or edit this list**, or edit it at the user level — no code change,
  no rebuild.
* This file holds **model ids and tuning only** — never URLs or keys. Those come from
  `~/.pi/agent/models.json` (or env overrides), and stay out of git.
* Search order: `$JEV_CONFIG` → `<extension dir>/jev-adapter.config.json` →
  `~/.pi/agent/jev-adapter.json`. The first existing file wins (no deep merge); copy the
  repo file to the user path for a personal override. An explicit `$JEV_CONFIG` that is
  missing or malformed is a **hard error** — silently deciding with the wrong model is worse.
* Field layering, per field: call opts → `JEV_*` env → matched entry → `defaults` → built-in.
* Config typos are surfaced, never swallowed: unknown tuning keys, bad types, and a default
  effort that is not in its own `supportedEfforts` all show up as `CONFIG PROBLEMS` in
  `/jev-config` (and the entry is still used).

## Endpoint resolution

1. `JEV_BASE_URL` / `JEV_MODEL` / `JEV_API_KEY` env vars, if set
2. otherwise the first `decisionModels` entry that matches a model in
   `~/.pi/agent/models.json` (that model's own provider supplies the `baseUrl` + key —
   `JEV_MODEL` never borrows another provider's credentials)
3. otherwise the tool fails with an error naming both what the config wants and what
   `models.json` actually serves — **no endpoint is hard-coded**

Other env: `JEV_CONFIG` (alternate config file), `JEV_MAX_TOKENS`, `JEV_TIMEOUT_MS`,
`JEV_RETRIES`, `JEV_MODELS_JSON`, `JEV_REASONING_EFFORT`, `JEV_RESPONSE_FORMAT`
(`json_object` default; `none` to disable).

## `reasoning_effort`: what actually buys what

Effort vocabulary, default, and latency are **per endpoint** — they live in the config
entry, and the numbers are rendered into the `jev_decide` tool description at load time
(run `/jev-config` to see them). Current defaults: `low`.

Measured on `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (vLLM, temp 0,
`response_format=json_object`, single state × 2 questions, medians of 3):

| effort | wall clock | reasoning tokens | note |
|---|---|---|---|
| `none` | **0.8–1.1 s** | 0 | the only latency switch (~10×); clean JSON |
| `low` | 8–10 s | ~420–500 | default |
| `medium` | 10–12 s | ~510–610 | |
| `xhigh` | 9–13 s | ~440–600 | server default when the field is omitted |
| `minimal` / `high` / `max` | — | — | **rejected** (`HTTP 400`); clamped via `effortMap` |

Two things worth knowing before you pick:

* Effort buys **no latency** on this server except `none`. `low` ≈ `medium` ≈ `xhigh` ≈ default.
* `none` **sharpens the distribution**. On a deliberately ambiguous state the top-candidate
  confidence was 0.70 vs 0.29 with thinking, and `P(need_human)` 0.15–0.25 vs 0.63–0.72 —
  i.e. the low-confidence "ask a human" signal partly disappears. Use `none` for cheap
  high-volume classification, `low`/`medium` when the caller routes on the confidence number.

Decode is the cost driver (~55 tok/s measured): a 10-state × 3-question batch was 18 s at
`none` and 37 s at `medium`, with all 10 states and all questions answered. Keep batches at
≤10 states × ≤6 questions, and remember the whole batch is **one** request.

A model with no config entry is still usable: if the server 400s on our `reasoning_effort`
the adapter drops the field, retries immediately, and reports `effortDropped`.

`response_format: json_object` (default) asks vLLM for syntactically-valid JSON — it removes
the parse-retry tier at zero latency/behaviour cost. Do **not** set `json_schema`: constrained
decoding at temp 0 collapses every answer to one-hot argmax and cannot express sum-to-1.

Each result reports `model`, `effort`, `elapsed`, `attempts` (requests actually made — >1 means
a retry rescued the call) and token `usage` in its `perf:` line.

## Inspecting / verifying the setup

```
/jev-config          # resolved baseURL/model/effort/tuning, config file used, candidates, typos
/jev-config check    # the same, plus one live decision round trip
```

```
node tests/jev-adapter/unit-config-resolution.mjs   # config/preference/clamp logic, mocked fetch
node tests/jev-adapter/unit-missing-questions.mjs   # partial-answer accounting, mocked fetch
```

## Usage

Ask the agent to make atomic judgements; it batches them into one `jev_decide` call:

```
Classify these 12 support tickets: department + urgency + customer frustration.
```

**Batch mode (multiple states):** pass `states: [{id, state}, ...]` to judge many
independent states (commits, tickets, diffs) against the SAME questions in one call —
answers come back keyed by state id. One batched call beats N parallel calls (the local
endpoint serializes on the GPU, so parallel calls just queue).

| Question type | You provide | You get back |
|---|---|---|
| `choice` | options map (≤255) | picked option + probability distribution + confidence |
| `score` | ordered levels (2–10) | fractional score + per-level probabilities + confidence |
| `boolean` | a statement | P(true) |

Route on confidence: `≥0.85 → act`, `0.5–0.85 → draft + human confirm`, `<0.5 → escalate`.
Tune thresholds to the risk of the action (and do not pick `effort: none` for the routing
signal — see the table above).

## Security notes

- **No secrets in this repo.** `jev-adapter.config.json` carries model ids and tuning; the
  endpoint URL/key come from your own `~/.pi/agent/models.json` (or env vars) at runtime and
  are never logged, rendered, or sent to the primary model (`/jev-config` masks the key).
- **The resolved `apiKey` is sent as a Bearer header to whatever `baseURL` resolves to.**
  Only point `JEV_BASE_URL` (or a models.json provider) at endpoints you trust — a hostile
  endpoint receives both the key and the judged state payload.
- **Prompt injection is inherent to the paradigm.** `state`, `instructions`, and
  `criteria` are model-controlled and interpolated into the decision prompt; a
  compromised primary agent could craft input that coaxes high confidence out of the
  decision model. Probabilities are **self-reported estimates, not calibrated** — every
  result carries this caveat. Keep a human in the loop for high-risk actions.
- The extension performs network calls to the configured decision endpoint only.
  No `eval`, no child processes, no filesystem writes; the only files read are
  `jev-adapter.config.json` and `models.json` (contents are never echoed).

## Credits

Ported from `dsh-jev-adapter` by BetterZflyee (MIT) — honesty prompt, reasoning-model
tolerance, probability normalisation, and retry behaviour are all from the original.
