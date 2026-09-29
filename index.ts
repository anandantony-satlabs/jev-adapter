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
	decide,
	decideBatch,
	describeJevConfig,
	resolveConfig,
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
		"To judge MULTIPLE independent states (commits, tickets, diffs) against the SAME questions, pass `states` instead of `state` — one batched call beats N parallel calls (the local endpoint serializes on the GPU). Output decode scales with states×questions, so keep both lean; ideal: ≤10 states × ≤6 questions.",
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
						"Batch mode: judge MULTIPLE independent states against the SAME questions in ONE call (beats N parallel tool calls; the endpoint serializes on the GPU). Mutually exclusive with `state`. Keep it lean: ≤10 states × ≤6 questions.",
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
		if (result.elapsedMs !== undefined) meta.push(`elapsed=${(result.elapsedMs / 1000).toFixed(1)}s`);
		if (result.attempts !== undefined && result.attempts > 1) meta.push(`attempts=${result.attempts} (retried)`);
		if (result.usage) meta.push(`tokens in/out=${result.usage.inputTokens}/${result.usage.outputTokens}`);
		if (meta.length) lines.push(`perf: ${meta.join("  ")}`);

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

			// "/jev-config check" — one cheap decision, so switching models in JSON is
			// verifiable without waiting for the agent to happen to call the tool.
			if (String(args ?? "").trim().toLowerCase() === "check") {
				if (!cfg) {
					lines.push("check: skipped (not configured)");
				} else {
					try {
						const t0 = Date.now();
						const ping = await decide(
							{ reasoningEffort: "none", maxTokens: 64 },
							{ probe: "connectivity check" },
							{ alive: { type: "boolean", instructions: "Is the decision endpoint answering normally?" } },
						);
						const p = (ping.answers as { alive?: { probability?: number } }).alive?.probability;
						lines.push(`check: OK — ${cfg.model} answered in ${((Date.now() - t0) / 1000).toFixed(1)}s (P(alive)=${p})`);
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
}
