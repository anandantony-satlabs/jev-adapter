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
 */

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { decide, loadEndpointFromModelsJson, resolveConfig, type JevQuestion } from "./adapter.js";

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
		"Each question must be one atomic, single-dimension judgement; compose complex logic from the answers yourself.",
		"choice and score answers include probabilities and a confidence score (distribution shape, 0-1, NOT a correctness guarantee);",
		"boolean answers return probability = P(true) with no confidence.",
		"Route on risk: higher-risk actions need higher confidence; low confidence goes to a human.",
	].join("\n"),
	parameters: Type.Object({
		state: Type.String({
			description:
				"The state / context to judge. A string, or a JSON-serialised object (preferred: name each relevant field — ticket, order, policy… — the decision model reads the state once and answers every question against it).",
		}),
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
	}),

	async execute(_toolCallId, params, signal, _onUpdate, _ctx) {
		const { state, questions } = params;
		if (!state) throw new Error("state is required");
		if (!questions || !Object.keys(questions).length) {
			throw new Error("questions is required (at least one question)");
		}

		const result = await decide({}, state, questions as Record<string, JevQuestion>, signal);

		// Compact model-facing text; full detail lives in `details`.
		const lines = [`channel: ${result.channel}`, `caveat: ${result.caveat}`, "answers:"];
		for (const [id, a] of Object.entries(result.answers)) {
			if (a.type === "boolean") {
				lines.push(`- ${id} [boolean] P(true)=${a.probability}`);
			} else {
				const top = Object.entries(a.probabilities ?? {})
					.sort((x, y) => y[1] - x[1])
					.slice(0, 3)
					.map(([k, v]) => `${k}=${v}`)
					.join(", ");
				lines.push(
					`- ${id} [${a.type}] ${a.type === "choice" ? a.choice : a.score} (confidence=${a.confidence}) top: ${top}`,
				);
			}
		}

		return {
			content: [{ type: "text", text: lines.join("\n") }],
			details: result,
		};
	},
});

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
				`baseURL:   ${cfg.baseURL}`,
				`model:     ${cfg.model}`,
				`apiKey:    ${cfg.apiKey ? "***" : "(none)"}`,
				`maxTokens: ${cfg.maxTokens}  timeout: ${cfg.timeoutMs}ms  retries: ${cfg.retries}`,
				`source:    ~/.pi/agent/models.json → ${fromModelsJson?.model ?? "(no match; using env overrides)"}`,
			].join("\n");
			if (ctx.hasUI) {
				ctx.ui.notify(text, "info");
			} else {
				console.log(text);
			}
		},
	});
}
