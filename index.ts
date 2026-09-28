/**
 * jev-adapter — pi extension port of dsh-jev-adapter (MIT, BetterZflyee).
 *
 * Registers one model tool, `jev_decide`, that runs the Jev (System One)
 * decision-model paradigm over the local OpenAI-compatible endpoint
 * declared in ~/.pi/agent/models.json (local-inference-lab/GLM-5.3-Flash*).
 *
 * Env overrides (all optional):
 *   JEV_BASE_URL / JEV_MODEL / JEV_API_KEY   point elsewhere
 *   JEV_MODELS_JSON                          alternate models.json path
 *   JEV_MAX_TOKENS / JEV_TIMEOUT_MS / JEV_RETRIES
 *   JEV_REASONING_EFFORT     none|minimal|low|medium|high|xhigh|max (default low) —
 *                            the latency control; hidden reasoning dominates wall time
 */

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
	decide,
	decideBatch,
	loadEndpointFromModelsJson,
	resolveConfig,
	type JevQuestion,
	type JevStateEntry,
} from "./adapter.js";

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
			Type.Array(
				Type.Object({
					id: Type.String({ description: "Unique state key (used in the answers) — e.g. a commit hash or ticket id" }),
					state: Type.Union([Type.String(), Type.Record(Type.String(), Type.Unknown())], {
						description: "The state to judge (string or JSON object).",
					}),
				}),
				{
					description:
						"Batch mode: judge MULTIPLE independent states against the SAME questions in ONE call (beats N parallel tool calls; the endpoint serializes on the GPU). Mutually exclusive with `state`. Keep it lean: ≤10 states × ≤6 questions.",
				},
			),
		),
		questions: Type.Record(
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
					description:
						"reasoning_effort for this call. Default 'low' (fast: hidden reasoning is the latency bottleneck — baseline ~36s vs ~1.2s at low). Use 'medium'/'high' for genuinely nuanced judgements. CAVEAT: 'none' does NOT disable thinking on GLM-5.3 servers — thinking leaks into content as prose; prefer 'low' for speed.",
				},
			),
		),
	}),

	async execute(_toolCallId, params, signal, _onUpdate, _ctx) {
		const { state, states, questions, effort } = params as {
			state?: unknown;
			states?: JevStateEntry[];
			effort?: string;
			questions?: Record<string, JevQuestion>;
		};
		if (!questions || !Object.keys(questions).length) {
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
			const batch = result as unknown as { answers: Record<string, Record<string, unknown>> };
			for (const [sid, qa] of Object.entries(batch.answers)) {
				lines.push(`state ${sid}:`, ...formatAnswers(qa).slice(1).map((l) => `  ${l}`));
			}
		} else {
			const single = result as unknown as { answers: Record<string, unknown> };
			lines.push(...formatAnswers(single.answers));
		}
		const meta: string[] = [];
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
		description: "Show the resolved jev_decide decision-model endpoint",
		handler: async (_args, ctx) => {
			const fromModelsJson = loadEndpointFromModelsJson();
			let cfg: ReturnType<typeof resolveConfig>;
			try {
				cfg = resolveConfig();
			} catch (e) {
				const msg = `jev-adapter not configured: ${(e as Error).message}`;
				if (ctx.hasUI) ctx.ui.notify(msg, "warning");
				else console.error(msg);
				return;
			}
			const text = [
				`baseURL:         ${cfg.baseURL}`,
				`model:           ${cfg.model}`,
				`apiKey:          ${cfg.apiKey ? "***" : "(none)"}`,
				`maxTokens:       ${cfg.maxTokens}  timeout: ${cfg.timeoutMs}ms  retries: ${cfg.retries}`,
				`reasoningEffort: ${cfg.reasoningEffort}`,
				`source:          ~/.pi/agent/models.json → ${fromModelsJson?.model ?? "(no match; using env overrides)"}`,
			].join("\n");
			if (ctx.hasUI) {
				ctx.ui.notify(text, "info");
			} else {
				console.log(text);
			}
		},
	});
}
