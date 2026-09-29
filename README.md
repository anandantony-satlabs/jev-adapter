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
      "match": "local-inference-lab/Qwen3.8-Flash-Next-NVFP4",
      "reasoningEffort": "low",
      "supportedEfforts": ["none", "low", "medium", "xhigh"],
      "effortMap": { "minimal": "low", "high": "xhigh", "max": "xhigh" },
      "effortGuide": "short how-to-pick text, rendered into the tool description (prompt tokens — keep it lean)",
      "note": "long-form measured behaviour of this endpoint — shown by /jev-config"
    },
    { "match": "local-inference-lab/GLM-5.3-Flash-NVFP4-Spark", "reasoningEffort": "low", "note": "fallback / rollback entry" }
  ]
}
```

* `decisionModels` is a **preference list**: the first entry whose `match` (exact id, or a
  substring of one) appears in a model id in `~/.pi/agent/models.json` wins. **Switching
  endpoints is `/jev-use` (below), not an edit** — the list itself rarely changes.
* `match` is pinned to the **exact id the server loads**, so one endpoint's tuning and effort
  vocabulary cannot leak onto another build of the same family (a non-Spark GLM, a newer Qwen
  revision). Use a substring when you *want* one entry to cover variants; add a separate entry
  when a variant behaves differently.
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

## Switching between two decision models

Switching is expected to happen back and forth (e.g.
`local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` ↔
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`, one GPU running one vLLM process), so it is
a command, not an edit:

```
/jev-use                 # list candidates + what the endpoint actually serves right now
/jev-use glm             # prefer the GLM entry (writes ~/.pi/agent/jev-adapter.local.json)
/jev-use qwen check      # switch back and run one live decision at that model's default effort
JEV_USE_MODEL=glm pi …   # same, for one process/session, without writing anything
```

* The preference only ever **reorders** `decisionModels` (`{ "prefer": ["…"] }` in the local
  file) — `jev-adapter.config.json` is never rewritten, so per-endpoint tuning and the effort
  vocabulary travel with the model. That matters because those two endpoints behave
  **oppositely**: on Qwen only `none` is fast (and it sharpens distributions), on GLM `low`
  is the fast switch and `none` leaks prose into `content`.
* `$JEV_USE_MODEL` beats the local file; unknown preference names are reported as `CONFIG
  PROBLEMS` and leave the order intact.
* `models.json` declares what *may* be served; `/jev-config` also probes `GET {baseURL}/models`
  to show what the endpoint serves *now*, and flags `← DOES NOT INCLUDE THE CONFIGURED MODEL`.
  If you switch while the other server is down, `jev_decide` fails with exactly that
  explanation (listing what is served and how to switch back) instead of an opaque 404.

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
/jev-config          # resolved baseURL/model/effort/tuning, config file used, candidates, typos,
                     # what models.json declares AND what the endpoint serves right now
/jev-config check    # the same, plus one live decision at the configured default effort
/jev-use             # current preference + candidates + live probe; /jev-use <name> switches
```

```
node tests/jev-adapter/unit-config-resolution.mjs   # config/preference/switch/clamp logic, mocked fetch
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
- The extension performs network calls to the configured decision endpoint only. No `eval`,
  no child processes. It reads `jev-adapter.config.json`, the local preference file, and
  `models.json` (contents never echoed), and **writes exactly one file**: the local
  preference file (`~/.pi/agent/jev-adapter.local.json`, path overridable with `$JEV_LOCAL`)
  and only when you run `/jev-use`. No secrets end up in any of them.

## Credits

Ported from `dsh-jev-adapter` by BetterZflyee (MIT) — honesty prompt, reasoning-model
tolerance, probability normalisation, and retry behaviour are all from the original.
