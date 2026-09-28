/**
 * The Jev adapter: turns the local OpenAI-compatible endpoint
 * (declared in ~/.pi/agent/models.json) into a System One style
 * decision engine.
 *
 * You send a state plus typed questions (choice / score / boolean);
 * it returns one calibrated-looking answer per question with
 * probabilities and a confidence score, in a single request.
 *
 * Ported from the dsh-jev-adapter Cordis plugin (MIT, BetterZflyee),
 * with the behaviours validated on real work items:
 *
 *  - prompts the model to spread probability honestly instead of
 *    collapsing to a fake 1.0 (without this, confidence saturates
 *    and the low-confidence routing signal is lost);
 *  - tolerates thinking models whose answer lands in reasoning_content
 *    (the local GLM 5.3 Flash endpoint is a reasoning model);
 *  - normalises probabilities that do not sum to 1;
 *  - retries with backoff on 429 / 5xx.
 */

import { readFileSync } from "node:fs";

/** Substring used to find the decision model inside ~/.pi/agent/models.json. */
const TARGET_MODEL_MATCH = "GLM-5.3-Flash";

// SECURITY NOTE: there is deliberately NO hard-coded endpoint fallback.
// The endpoint always comes from ~/.pi/agent/models.json (or JEV_BASE_URL /
// JEV_MODEL / JEV_API_KEY env overrides). Do not commit real hostnames,
// tailnet names, or keys into this file — it is meant for a public repo.

export interface JevQuestion {
	type: "choice" | "score" | "boolean";
	instructions?: string;
	criteria?: unknown;
}

export interface JevAdapterConfig {
	baseURL?: string;
	apiKey?: string;
	model?: string;
	maxTokens?: number;
	timeoutMs?: number;
	retries?: number;
}

export interface JevDecisionResult {
	channel: string;
	answers: Record<
		string,
		{
			type: "choice" | "score" | "boolean";
			choice?: string;
			score?: number;
			probability?: number;
			probabilities?: Record<string, number>;
			confidence?: number;
		}
	>;
	caveat: string;
	usage?: { inputTokens: number; outputTokens: number };
}

/* ------------------------------------------------------------------ */
/* Endpoint resolution from ~/.pi/agent/models.json                    */
/* ------------------------------------------------------------------ */

interface ModelsJson {
	providers?: Record<
		string,
		{
			baseUrl?: string;
			apiKey?: string;
			models?: { id?: string; name?: string }[];
		}
	>;
}

export interface EndpointInfo {
	baseURL: string;
	apiKey: string;
	model: string;
}

/** Locate the decision model in ~/.pi/agent/models.json (or a path override). Returns null when unreadable or unmatched. */
export function loadEndpointFromModelsJson(path?: string): EndpointInfo | null {
	let file = path ?? process.env.JEV_MODELS_JSON;
	if (!file) {
		const home = process.env.HOME ?? process.env.USER_PROFILE ?? "";
		file = `${home}/.pi/agent/models.json`;
	}
	try {
		const raw = JSON.parse(readFileSync(file)) as ModelsJson;
		const providers = raw.providers ?? {};
		for (const provider of Object.values(providers)) {
			for (const model of provider.models ?? []) {
				const id = model.id ?? "";
				if (id.includes(TARGET_MODEL_MATCH) && provider.baseUrl) {
					return {
						baseURL: provider.baseUrl.replace(/\/$/, ""),
						apiKey: provider.apiKey ?? "",
						model: id,
					};
				}
			}
		}
	} catch {
		/* unreadable or malformed models.json — treat as unresolved */
	}
	return null;
}

/* ------------------------------------------------------------------ */
/* Question helpers                                                    */
/* ------------------------------------------------------------------ */

interface Candidate {
	name: string;
	desc?: string;
}

/**
 * Normalise a question's criteria into a flat candidate list.
 * choice → [{name, desc}] from the criteria object/array
 * score  → [{name: '0', desc}, ...] ordered levels
 */
export function candidatesOf(q: JevQuestion): Candidate[] {
	const c = q.criteria;
	if (q.type === "choice") {
		if (Array.isArray(c)) return c.map((d, i) => ({ name: String(i), desc: d as string }));
		if (c && typeof c === "object")
			return Object.keys(c as Record<string, unknown>).map((k) => ({
				name: k,
				desc: (c as Record<string, unknown>)[k] as string,
			}));
		return [];
	}
	if (q.type === "score") {
		const arr = Array.isArray(c) ? c : [];
		return arr.map((d, i) => ({ name: String(i), desc: d as string }));
	}
	return [];
}

/**
 * Build the decision prompt. The honesty clause is deliberate:
 * without it models output hard 0/1 probabilities, confidence
 * saturates at 1.0, and the caller cannot tell "sure" from "guessing".
 */
export function buildPrompt(state: unknown, questions: Record<string, JevQuestion>): string {
	const lines = [
		"You are a decision scorer. Given the STATE, output a probability distribution over the candidates for every question.",
		"Rules:",
		"1. Probabilities must reflect your genuine judgement. Never spread them uniformly out of laziness. The probabilities of each question must sum to 1.",
		"2. If you are genuinely uncertain, spread the probability across multiple candidates honestly — do NOT force a fake 1.0 onto one option. Your uncertainty is itself a useful signal: the caller routes on it (auto / ask-a-human / escalate), and a dishonest 1.0 breaks that mechanism. Only assign near-1 probabilities when the evidence is unambiguous.",
		"3. Output ONLY JSON. No explanations, no markdown fences.",
		"4. Use the given candidate names exactly as JSON keys.",
		"",
		"── STATE ──",
		typeof state === "string" ? state : JSON.stringify(state, null, 2),
		"",
		"── QUESTIONS ──",
	];
	const shape: Record<string, unknown> = {};
	let i = 0;
	for (const id of Object.keys(questions)) {
		i += 1;
		const q = questions[id];
		lines.push(`${i}. ${id} [${q.type}] ${q.instructions ?? ""}`);
		if (q.type === "boolean") {
			lines.push("   Output your estimate of P(true), between 0 and 1.");
			shape[id] = { probability: 0.0 };
		} else {
			const cands = candidatesOf(q);
			lines.push(
				`   Candidates (${cands.length}): ` +
					cands.map((c) => `${c.name}=${c.desc ?? ""}`).join(" / "),
			);
			lines.push(
				q.type === "score"
					? "   Output the probability of each level (levels numbered from 0 in the order listed)."
					: "   Output the probability of each candidate.",
			);
			const probs: Record<string, number> = {};
			cands.forEach((c) => {
				probs[c.name] = 0.0;
			});
			shape[id] = { probabilities: probs };
		}
	}
	lines.push("", "── OUTPUT (this JSON only) ──", JSON.stringify(shape, null, 2));
	return lines.join("\n");
}

/** Extract the first JSON object from a raw model output. */
export function extractJson(text: string): Record<string, unknown> | null {
	if (!text) return null;
	let t = text.trim();
	const fence = t.match(/```(?:json)?\s*([\s\S]*?)```/i);
	if (fence) t = fence[1].trim();
	const s = t.indexOf("{");
	const e = t.lastIndexOf("}");
	if (s === -1 || e === -1 || e <= s) return null;
	try {
		return JSON.parse(t.slice(s, e + 1)) as Record<string, unknown>;
	} catch {
		return null;
	}
}

/**
 * Confidence of a distribution: how concentrated it is.
 * Uniform → 0, all mass on one candidate → 1.
 * This mirrors the shape of Jev's confidence statistic (not its
 * calibration — see the caveat attached to every result).
 */
export function confidenceOf(probs: number[], n: number): number {
	if (n <= 1) return 1;
	const maxP = Math.max(...probs);
	const c = (maxP - 1 / n) / (1 - 1 / n);
	return Math.max(0, Math.min(1, Number(c.toFixed(4))));
}

/* ------------------------------------------------------------------ */
/* Channel execution                                                   */
/* ------------------------------------------------------------------ */

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function numEnv(name: string): number | undefined {
	const v = process.env[name];
	if (!v) return undefined;
	const n = Number(v);
	return Number.isFinite(n) ? n : undefined;
}

/** Resolve the effective config: explicit opts → env vars → models.json. Throws when no endpoint can be resolved. */
export function resolveConfig(opts: JevAdapterConfig = {}): Required<JevAdapterConfig> {
	const fromModelsJson = loadEndpointFromModelsJson();
	const baseURL = (opts.baseURL ?? process.env.JEV_BASE_URL ?? fromModelsJson?.baseURL ?? "").replace(/\/$/, "");
	if (!baseURL) {
		throw new Error(
			`jev-adapter: no decision endpoint. Add a provider with a "${TARGET_MODEL_MATCH}" model to ~/.pi/agent/models.json, or set JEV_BASE_URL + JEV_MODEL (and JEV_API_KEY if the endpoint needs auth).`,
		);
	}
	// SECURITY: whatever baseURL resolves to, the resolved apiKey is sent to
	// it as a Bearer header. Only point JEV_BASE_URL at endpoints you trust —
	// a hostile endpoint receives both the key and the judged state payload.
	return {
		baseURL,
		apiKey: opts.apiKey ?? process.env.JEV_API_KEY ?? fromModelsJson?.apiKey ?? "",
		model: opts.model ?? process.env.JEV_MODEL ?? fromModelsJson?.model ?? "",
		maxTokens: opts.maxTokens ?? numEnv("JEV_MAX_TOKENS") ?? 4000,
		timeoutMs: opts.timeoutMs ?? numEnv("JEV_TIMEOUT_MS") ?? 120000,
		retries: opts.retries ?? numEnv("JEV_RETRIES") ?? 3,
	};
}

async function callOpenAiCompatible(
	cfg: Required<JevAdapterConfig>,
	prompt: string,
	signal?: AbortSignal,
): Promise<{ raw: string; usage?: { inputTokens: number; outputTokens: number } }> {
	let lastErr = "";
	for (let attempt = 0; attempt < cfg.retries; attempt += 1) {
		if (attempt) await sleep(1500 * 2 ** (attempt - 1));
		if (signal?.aborted) throw new Error("Aborted");
		let res: Response;
		try {
			res = await fetch(`${cfg.baseURL}/chat/completions`, {
				method: "POST",
				headers: {
					Authorization: `Bearer ${cfg.apiKey}`,
					"Content-Type": "application/json",
				},
				body: JSON.stringify({
					model: cfg.model,
					messages: [
						{ role: "system", content: "You are a rigorous decision scorer. Output JSON only." },
						{ role: "user", content: prompt },
					],
					max_tokens: cfg.maxTokens,
					temperature: 0,
				}),
				signal: signal
					? AbortSignal.any([signal, AbortSignal.timeout(cfg.timeoutMs)])
					: AbortSignal.timeout(cfg.timeoutMs),
			});
		} catch (e) {
			if (signal?.aborted) throw new Error("Aborted");
			lastErr = `network error: ${(e as Error).message}`;
			continue;
		}
		if (res.status === 429 || res.status >= 500) {
			lastErr = `HTTP ${res.status} (retrying)`;
			continue;
		}
		if (!res.ok) {
			lastErr = `HTTP ${res.status}: ${(await res.text()).slice(0, 250)}`;
			break;
		}
		const data = (await res.json()) as {
			choices?: { message?: { content?: string; reasoning_content?: string } }[];
			usage?: { prompt_tokens?: number; completion_tokens?: number };
		};
		// Thinking models often leave content empty and put the answer
		// in reasoning_content; read both.
		const msg = data.choices?.[0]?.message ?? {};
		return {
			raw: (msg.content && String(msg.content).trim()) || msg.reasoning_content || "",
			usage: data.usage
				? {
						inputTokens: data.usage.prompt_tokens ?? 0,
						outputTokens: data.usage.completion_tokens ?? 0,
					}
				: undefined,
		};
	}
	throw new Error(`decision endpoint failed (${cfg.model}): ${lastErr}`);
}

/* ------------------------------------------------------------------ */
/* Result assembly                                                     */
/* ------------------------------------------------------------------ */

function assembleAnswers(
	questions: Record<string, JevQuestion>,
	parsed: Record<string, unknown>,
): JevDecisionResult["answers"] {
	const answers: JevDecisionResult["answers"] = {};
	for (const id of Object.keys(questions)) {
		const q = questions[id];
		const a = parsed[id] as
			| { probability?: unknown; p?: unknown; probabilities?: Record<string, unknown> }
			| undefined;
		if (!a) continue;
		if (q.type === "boolean") {
			const p = Number(a.probability ?? a.p);
			if (!Number.isFinite(p)) continue;
			const clamped = Math.max(0, Math.min(1, p));
			answers[id] = { type: "boolean", probability: Number(clamped.toFixed(4)) };
		} else {
			const cands = candidatesOf(q);
			if (!cands.length) continue;
			let probs = cands.map((c) => {
				const v = Number(a.probabilities?.[c.name]);
				return Number.isFinite(v) && v >= 0 ? v : 0;
			});
			const sum = probs.reduce((x, y) => x + y, 0);
			if (sum <= 0) {
				// All-zero answer degrades to uniform rather than failing.
				probs = cands.map(() => 1 / cands.length);
			} else {
				probs = probs.map((v) => v / sum);
			}
			const norm: Record<string, number> = {};
			cands.forEach((c, k) => {
				norm[c.name] = Number(probs[k].toFixed(4));
			});
			const conf = confidenceOf(probs, cands.length);
			if (q.type === "choice") {
				const best = probs.indexOf(Math.max(...probs));
				answers[id] = {
					type: "choice",
					choice: cands[best].name,
					probabilities: norm,
					confidence: conf,
				};
			} else {
				const score = probs.reduce((acc, p, k) => acc + p * Number(cands[k].name), 0);
				answers[id] = {
					type: "score",
					score: Number(score.toFixed(4)),
					probabilities: norm,
					confidence: conf,
				};
			}
		}
	}
	return answers;
}

/**
 * Run one decision request.
 *
 * @param opts  adapter config (baseURL / apiKey / model / maxTokens / …)
 * @param state the state to judge
 * @param questions question map: questionId → {type, instructions, criteria}
 * @param signal optional abort signal from the calling tool
 */
export async function decide(
	opts: JevAdapterConfig,
	state: unknown,
	questions: Record<string, JevQuestion>,
	signal?: AbortSignal,
): Promise<JevDecisionResult> {
	const cfg = resolveConfig(opts);
	if (!cfg.model) {
		throw new Error("jev-adapter: no decision model configured (set JEV_MODEL or add one to ~/.pi/agent/models.json).");
	}
	const prompt = buildPrompt(state, questions);
	const { raw, usage } = await callOpenAiCompatible(cfg, prompt, signal);
	const parsed = extractJson(raw);
	if (!parsed) {
		throw new Error(
			`Could not parse JSON from model output (first 200 chars: ${String(raw).slice(0, 200)})`,
		);
	}
	const answers = assembleAnswers(questions, parsed);
	if (!Object.keys(answers).length) {
		throw new Error("Model output JSON matched none of the question IDs");
	}
	return {
		channel: `local decision model (${cfg.model})`,
		answers,
		caveat:
			"Probabilities are the model's self-reported estimates, not mathematically calibrated. Review high-risk decisions manually.",
		usage,
	};
}
