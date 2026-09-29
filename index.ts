/**
 * jev-adapter — pi extension port of dsh-jev-adapter (MIT, BetterZflyee).
 *
 * Registers one model tool, `jev_decide`, that runs the Jev (System One)
 * decision-model paradigm over a local OpenAI-compatible endpoint.
 *
 * WHICH model decides is data, not code: it comes from
 * jev-adapter.config.json (`decisionModels`, matched against the model ids in
 * ~/.pi/agent/models.json). Pointing the tool at a different model is an edit
 * to that JSON file — nothing in this file names a model.
 *
 * Env overrides (all optional):
 *   JEV_CONFIG                               alternate jev-adapter.config.json
 *   JEV_BASE_URL / JEV_MODEL / JEV_API_KEY   point elsewhere
 *   JEV_MODELS_JSON                          alternate models.json path
 *   JEV_MAX_TOKENS / JEV_TIMEOUT_MS / JEV_RETRIES
 *   JEV_REASONING_EFFORT     default effort (a per-call `effort` beats it; the
 *                            config entry's supportedEfforts/effortMap decide what
 *                            is actually sendable — unsupported values are clamped,
 *                            never sent, and the clamp is reported back)
 *   JEV_RESPONSE_FORMAT      json_object (default; server-guaranteed valid JSON) | none
 */

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
	adapterLocalPath,
	decide,
	decideBatch,
	type JevBudgetEstimate,
	describeJevConfig,
	listServedModels,
	nameMatches,
	resolveConfig,
	writeModelPreference,
	type JevQuestion,
	type JevStateEntry,
} from "./adapter.js";

/**
 * Effort vocabulary + how to choose it on the *currently configured* decision
 * model, rendered into the tool description so the driving model sees real,
 * config-owned numbers instead of a stale hard-coded guess. Deliberately short
 * (this text is in every prompt); the long measured note stays in /jev-config.
 * Best-effort: if the config cannot be resolved the extension still loads.
 */
function effortHint(): string {
	try {
		const r = resolveConfig();
		const cand = describeJevConfig().candidates.find((c) => c.match === r.matchedBy);
		const supported = cand?.supportedEfforts?.length ? cand.supportedEfforts.join(" | ") : "endpoint-dependent";
		const guide =
			cand?.effortGuide ??
			"No measured effort guide for this endpoint in jev-adapter.config.json — run /jev-config to see what it accepts.";
		return `Effort: choose from ${supported} (anything else is clamped per jev-adapter.config.json and reported as effortClampedFrom). Default here: "${r.effortApplied || "(server default)"}". ${guide}`;
	} catch {
		return "Effort vocabulary and latency are endpoint-specific — run /jev-config to see what the configured decision model accepts and how fast each level is.";
	}
}

const jevDecideTool = defineTool({
	name: "jev_decide",
	label: "Jev Decide",
	description: [
		"Run a structured decision over one state: send typed questions (choice / score / boolean) and get back probability-distributed answers in a single call.",
		"",
		"Good for high-frequency atomic judgements: classification, routing, scoring, fact-checking, dedup, quality-gating another model's output.",
		"Not for text generation, explanations, or multi-step reasoning — use a normal model for those.",
		"",
		"You may ask many questions in one call; 1 and 10 questions cost about the same, so batch every judgement you might need.",
		"To judge MULTIPLE independent states (commits, tickets, diffs) against the SAME questions, pass `states` instead of `state` — one batched call beats N parallel calls (the local endpoint serializes on the GPU). Output decode scales with states×questions: budget ~280 output tokens per score answer, ~160 per choice, ~130 per boolean, and the whole answer set must fit in one response (max_tokens) or it comes back unparseable. Measured ceiling with rich (50+ word) states: ~6 states × 3 questions at low effort; ~10 × 3 only if the answer total stays under max_tokens.",
		"Each question must be one atomic, single-dimension judgement; compose complex logic from the answers yourself.",
		"choice and score answers include probabilities and a confidence score (distribution shape, 0-1, NOT a correctness guarantee);",
		"boolean answers return probability = P(true) with no confidence.",
		"Route on risk: higher-risk actions need higher confidence; low confidence goes to a human.",
	].join("\n"),
	parameters: Type.Object({
		state: Type.Optional(
			Type.Union([Type.String(), Type.Record(Type.String(), Type.Unknown())], {
				description:
					"The state / context to judge. A string, or a JSON-serialised object (preferred: name each relevant field — ticket, order, policy… — the decision model reads the state once and answers every question against it). Omit when using `states` batch mode.",
			}),
		),
		states: Type.Optional(
			// Union with Type.String(): driving models sometimes emit a nested array
			// as a JSON-encoded string (a tool-args quirk that grows with payload
			// size); pi validates before execute(), so a strict array schema turns
			// that into a hard validation error. Accept the string form and
			// JSON.parse it in execute() instead.
			Type.Union(
				[
					Type.Array(
						Type.Object({
							id: Type.String({ description: "Unique state key (used in the answers) — e.g. a commit hash or ticket id" }),
							state: Type.Union([Type.String(), Type.Record(Type.String(), Type.Unknown())], {
								description: "The state to judge (string or JSON object).",
							}),
						}),
					),
					Type.String(),
				],
				{
					description:
						"Batch mode: judge MULTIPLE independent states against the SAME questions in ONE call (beats N parallel tool calls; the endpoint serializes on the GPU). Mutually exclusive with `state`. Budget the OUTPUT, not just the count: states × (sum of per-question decode costs: score ~280, choice ~160, boolean ~130 tokens) must fit under max_tokens, so ~6 states × 3 questions with rich state text is the safe shape.",
				},
			),
		),
		questions: Type.Union([
			Type.Record(
			Type.String({ description: "questionId — only used to key the answers" }),
			Type.Object({
				type: Type.Union([Type.Literal("choice"), Type.Literal("score"), Type.Literal("boolean")], {
					description: "Kind of atomic judgement.",
				}),
				instructions: Type.Optional(
					Type.String({
						description:
							"What to judge — direct, single-dimension wording works best.",
					}),
				),
				criteria: Type.Optional(
					Type.Unknown({
						description:
							"choice: object of option→description (up to 255 options) or array of descriptions; score: array of level descriptions, low→high, 2-10 levels; boolean: omit.",
					}),
				),
			}),
			),
			// Same model-quirk tolerance as `states`: accept a JSON-encoded string.
			Type.String(),
		]),
		effort: Type.Optional(
			Type.Union(
				[
					Type.Literal("none"),
					Type.Literal("minimal"),
					Type.Literal("low"),
					Type.Literal("medium"),
					Type.Literal("high"),
					Type.Literal("xhigh"),
					Type.Literal("max"),
				],
				{
					description: `reasoning_effort for this call. ${effortHint()}`,
				},
			),
		),
	}),

	async execute(_toolCallId, params, signal, _onUpdate, _ctx) {
		// Model-quirk tolerance: driving models sometimes emit nested JSON (the
		// states array, the questions record) as a JSON-encoded string. pi
		// validates against the widened schema (string | structure) and hands
		// both forms through; coerce here so downstream code sees structures.
		const parseMaybe = (v: unknown): unknown => {
			if (typeof v !== "string") return v;
			const s = v.trim();
			if (!(s.startsWith("{") || s.startsWith("["))) return v;
			try {
				return JSON.parse(s);
			} catch {
				return v; // not valid JSON — keep as a plain string state
			}
		};
		const state = parseMaybe(params.state) as unknown;
		const statesParsed = parseMaybe(params.states) as JevStateEntry[] | string | undefined;
		const questions = parseMaybe(params.questions) as Record<string, JevQuestion> | string | undefined;
		const effort = (params as { effort?: string }).effort;
		const states = Array.isArray(statesParsed) ? statesParsed : undefined;
		if (!questions || typeof questions !== "object" || Array.isArray(questions) || !Object.keys(questions).length) {
			throw new Error("questions is required (at least one question)");
		}
		if (!state && !(states && states.length)) {
			throw new Error("state is required (or pass `states` for batch mode)");
		}
		if (state && states && states.length) {
			throw new Error("Pass either `state` or `states`, not both");
		}

		// resolveConfig() reads JEV_REASONING_EFFORT (and all other JEV_* envs) itself;
		// a per-call `effort` overrides it.
		const result = states && states.length
			? await decideBatch({ reasoningEffort: effort ?? undefined }, states, questions, signal)
			: await decide(
					{ reasoningEffort: effort ?? undefined },
					state,
					questions as Record<string, JevQuestion>,
					signal,
				);

		// Compact model-facing text; full detail lives in `details`.
		const lines = [`channel: ${result.channel}`, `caveat: ${result.caveat}`];
		if (states && states.length) {
			const batch = result as unknown as {
				answers: Record<string, Record<string, unknown>>;
				missingStates?: string[];
				missingQuestions?: Record<string, string[]>;
			};
			for (const [sid, qa] of Object.entries(batch.answers)) {
				lines.push(`state ${sid}:`, ...formatAnswers(qa).slice(1).map((l) => `  ${l}`));
			}
			if (batch.missingStates?.length) {
				lines.push(`WARNING: no answer for state(s): ${batch.missingStates.join(", ")}`);
			}
			if (batch.missingQuestions && Object.keys(batch.missingQuestions).length) {
				const parts = Object.entries(batch.missingQuestions).map(
					([qid, sids]) => `${qid} (${sids.length}/${states.length} states)`,
				);
				lines.push(`WARNING: question(s) silently dropped by some state(s): ${parts.join(", ")} - re-ask or treat results as partial`);
			}
		} else {
			const single = result as unknown as { answers: Record<string, unknown>; missingQuestions?: string[] };
			lines.push(...formatAnswers(single.answers));
			if (single.missingQuestions?.length) {
				lines.push(`WARNING: question(s) silently dropped: ${single.missingQuestions.join(", ")} - re-ask or treat results as partial`);
			}
		}
		const meta: string[] = [];
		if (result.model) meta.push(`model=${result.model}`);
		if (result.reasoningEffort !== undefined) meta.push(`effort=${result.reasoningEffort || "(server default)"}`);
		if (result.effortClampedFrom) {
			lines.push(
				`WARNING: reasoning_effort "${result.effortClampedFrom}" is not sendable to this endpoint → sent "${result.reasoningEffort || "(nothing; server default)"}". Add the mapping to jev-adapter.config.json if that is not what you meant.`,
			);
		}
		if (result.modelFallbackFrom) {
			lines.push(
				`WARNING: preferred decision model "${result.modelFallbackFrom}" was unavailable (not declared in ~/.pi/agent/models.json) → ${result.model} answered, which has a DIFFERENT effort vocabulary and latency. Start the other server (declare its model in models.json) or run /jev-use <available>.`,
			);
		}
		if (result.elapsedMs !== undefined) meta.push(`elapsed=${(result.elapsedMs / 1000).toFixed(1)}s`);
		if (result.attempts !== undefined && result.attempts > 1) meta.push(`attempts=${result.attempts} (retried)`);
		if (result.usage) meta.push(`tokens in/out=${result.usage.inputTokens}/${result.usage.outputTokens}`);
		const budget = (result as unknown as { budget?: JevBudgetEstimate }).budget;
		const raised = (result as unknown as { maxTokensRaisedFrom?: number }).maxTokensRaisedFrom;
		// est_out next to the real out-token count keeps the per-answer decode costs
		// in estimateOutputBudget() honest every time the tool is called.
		if (budget) meta.push(`est_out=${budget.estOutputTokens}`);
		if (meta.length) lines.push(`perf: ${meta.join("  ")}`);
		if (raised) {
			lines.push(
				`WARNING: an attempt was cut off at ${raised} output tokens and retried with a raised max_tokens. The batch was too big for one response - shrink it (fewer states, or fewer score questions) rather than relying on the rescue: each raise costs a whole extra decode pass.`,
			);
		} else if (budget && budget.chunks > 1) {
			lines.push(
				`WARNING: ~${budget.estOutputTokens} output tokens needed for ${budget.answers} answers but max_tokens=${budget.maxTokens} - split into ~${budget.chunks} calls. A cut-off answer is not partially usable: the JSON will not parse and every retry fails the same way.`,
			);
		}

		return {
			content: [{ type: "text", text: lines.join("\n") }],
			details: result,
		};
	},
});

/** Compact model-facing answer lines for one answer map. */
function formatAnswers(answers: Record<string, unknown>): string[] {
	const out: string[] = ["answers:"];
	for (const [id, a] of Object.entries(answers)) {
		const ans = a as { type: string; probability?: number; choice?: string; score?: number; confidence?: number; probabilities?: Record<string, number> };
		if (ans.type === "boolean") {
			out.push(`- ${id} [boolean] P(true)=${ans.probability}`);
		} else {
			const top = Object.entries(ans.probabilities ?? {})
				.sort((x, y) => y[1] - x[1])
				.slice(0, 3)
				.map(([k, v]) => `${k}=${v}`)
				.join(", ");
			out.push(
				`- ${id} [${ans.type}] ${ans.type === "choice" ? ans.choice : ans.score} (confidence=${ans.confidence}) top: ${top}`,
			);
		}
	}
	return out;
}

export default function (pi: ExtensionAPI) {
	pi.registerTool(jevDecideTool);

	pi.registerCommand("jev-config", {
		description: "Show the resolved jev_decide decision model (add \"check\" for a live round trip)",
		handler: async (args, ctx) => {
			const report = describeJevConfig();
			const cfg = report.resolved;
			const lines: string[] = [];
			if (cfg) {
				lines.push(
					`baseURL:         ${cfg.baseURL}`,
					`model:           ${cfg.model}${cfg.matchedBy ? `  (selected by "${cfg.matchedBy}")` : ""}`,
					`apiKey:          ${cfg.apiKey ? "***" : "(none)"}`,
					`maxTokens:       ${cfg.maxTokens}  timeout: ${cfg.timeoutMs}ms  retries: ${cfg.retries}`,
					`reasoningEffort: ${cfg.effortApplied || "(server default)"}${cfg.effortRequested ? `  (requested "${cfg.effortRequested}" → clamped)` : ""}  responseFormat: ${cfg.responseFormat}`,
					`config:          ${cfg.configPath ?? "(built-in defaults — no jev-adapter.config.json found)"}`,
				);
			} else {
				lines.push(
					`jev-adapter not configured: ${report.error ?? "unknown error"}`,
					`searched for config: ${report.searchOrder.join(", ")}`,
				);
			}
			lines.push(`models.json serves: ${report.served.length ? report.served.join(", ") : "(nothing readable)"}`);
			if (cfg) {
				// Declared vs actually loaded: one GPU usually runs one decision model,
				// and models.json only declares what MAY be there.
				const live = await listServedModels(cfg);
				lines.push(
					`endpoint serves now: ${
						live.ok ? (live.models.length ? live.models.join(", ") : "(nothing loaded)") : `(probe failed: ${live.error})`
					}${live.ok && live.models.length > 0 && !live.models.includes(cfg.model) ? "   ← DOES NOT INCLUDE THE CONFIGURED MODEL" : ""}`,
				);
				lines.push(
					`preference:      ${
						cfg.preferred?.length
							? `${cfg.preferred.join(" > ")} (from ${process.env.JEV_USE_MODEL ? "$JEV_USE_MODEL" : adapterLocalPath()})`
							: "file order (no override)"
					} — switch with /jev-use <name>${cfg.preferredNotServed ? `   ← preferred "${cfg.preferredNotServed}" NOT declared in models.json; using the next entry` : ""}`,
				);
			}
			lines.push(
				`configured candidates (first match wins): ${
					report.candidates.length
						? report.candidates
								.map(
									(c) =>
										`${c.selected ? "[x]" : "[ ]"} ${c.match}${c.supportedEfforts?.length ? ` (effort ${c.supportedEfforts.join("|")})` : ""}`,
								)
								.join(", ")
						: "(none — set JEV_MODEL or edit jev-adapter.config.json)"
				}`,
			);
			if (cfg?.profileNote) lines.push(`note: ${cfg.profileNote}`);
			if (report.problems.length) lines.push(`CONFIG PROBLEMS: ${report.problems.join(" | ")}`);

			// "/jev-config check" — one real decision at the CONFIG DEFAULT effort, so
			// what gets verified is the path the tool will actually take (forcing
			// effort=none would verify a special case and falsely fail on endpoints
			// where 'none' leaks prose into content).
			if (String(args ?? "").trim().toLowerCase() === "check") {
				if (!cfg) {
					lines.push("check: skipped (not configured)");
				} else {
					try {
						const t0 = Date.now();
						const ping = await decide(
							{ maxTokens: 128 },
							{ probe: "connectivity check" },
							{ alive: { type: "boolean", instructions: "Is the decision endpoint answering normally?" } },
						);
						const p = (ping.answers as { alive?: { probability?: number } }).alive?.probability;
						lines.push(
							`check: OK — ${cfg.model} answered in ${((Date.now() - t0) / 1000).toFixed(1)}s at effort "${ping.reasoningEffort || "(server default)"}" (P(alive)=${p})`,
						);
					} catch (e) {
						lines.push(`check: FAILED — ${(e as Error).message}`);
					}
				}
			}
			const text = lines.join("\n");
			if (ctx.hasUI) {
				ctx.ui.notify(text, report.error || report.problems.length ? "warning" : "info");
			} else {
				console.log(text);
			}
		},
	});

	// Endpoint ping-pong (GLM-5.3-Flash ↔ Qwen3.8-Next, or anything else declared in
	// the config): one command, no editor. The effort vocabulary/guide switches with
	// the model because both are per-entry in jev-adapter.config.json. The choice is
	// written to a local, un-versioned file, so the tracked config file stays clean.
	pi.registerCommand("jev-use", {
		description: "Switch the decision model: /jev-use <name|substring> [check] — with no argument, list candidates and probe the endpoint live",
		handler: async (args, ctx) => {
			const parts = String(args ?? "").trim().split(/\s+/).filter(Boolean);
			const want = parts.find((p) => p.toLowerCase() !== "check") ?? "";
			const alsoCheck = parts.some((p) => p.toLowerCase() === "check");
			const lines: string[] = [];
			const emit = (warn: boolean) => {
				if (ctx.hasUI) ctx.ui.notify(lines.join("\n"), warn ? "warning" : "info");
				else console.log(lines.join("\n"));
			};

			if (want) {
				const before = describeJevConfig();
				const hit = before.candidates.find((c) => nameMatches(c.match, want));
				if (!hit) {
					lines.push(
						`"${want}" matches no decisionModels entry. Configured: ${before.candidates.map((c) => c.match).join(", ") || "(none)"}.`,
						`Add it to ${before.configPath ?? "jev-adapter.config.json"} first (match + supportedEfforts + effortGuide) — then /jev-use will work.`,
					);
					emit(true);
					return;
				}
				lines.push(`preference saved → ${hit.match} (${writeModelPreference(hit.match)})`);
			}

			const report = describeJevConfig();
			const cfg = report.resolved;
			if (!cfg) {
				lines.push(`not configured: ${report.error ?? "unknown error"}`);
				emit(true);
				return;
			}
			const cand = report.candidates.find((c) => c.match === cfg.matchedBy);
			if (cfg.preferredNotServed) {
				lines.push(
					`NOTICE: preferred "${cfg.preferredNotServed}" is not declared in ~/.pi/agent/models.json, so "${cfg.matchedBy}" answers instead — DIFFERENT effort vocabulary and latency. Start the other server (and declare its model in models.json), or /jev-use <available>.`,
				);
			}
			lines.push(
				`using:    ${cfg.model}`,
				`effort:   default "${cfg.effortApplied || "(server default)"}", sendable ${cand?.supportedEfforts?.join("|") ?? "(endpoint-dependent)"}`,
				`          ${cand?.effortGuide ?? ""}`,
			);
			const live = await listServedModels(cfg);
			if (!live.ok) {
				lines.push(`live probe: FAILED (${live.error}) — endpoint unreachable?`);
			} else if (!live.models.includes(cfg.model)) {
				lines.push(
					`live probe: the endpoint serves ${live.models.length ? live.models.join(", ") : "(nothing)"} — NOT ${cfg.model}. Preference is saved, but decisions will fail until that server is up (or /jev-use <other>).`,
				);
			} else {
				lines.push(`live probe: OK — endpoint serves ${cfg.model}`);
			}
			if (alsoCheck) {
				try {
					const t0 = Date.now();
					const ping = await decide({ maxTokens: 128 }, { probe: "switch verification" }, {
						alive: { type: "boolean", instructions: "Is the decision endpoint answering normally?" },
					});
					lines.push(
						`check: OK — answered in ${((Date.now() - t0) / 1000).toFixed(1)}s at effort "${ping.reasoningEffort || "(server default)"}" (P(alive)=${(ping.answers as { alive?: { probability?: number } }).alive?.probability})`,
					);
				} catch (e) {
					lines.push(`check: FAILED — ${(e as Error).message}`);
				}
			} else {
				lines.push('tip: add "check" to also run one live decision at this model\'s default effort');
			}
			if (report.problems.length) lines.push(`CONFIG PROBLEMS: ${report.problems.join(" | ")}`);
			emit(report.problems.length > 0);
		},
	});
}
