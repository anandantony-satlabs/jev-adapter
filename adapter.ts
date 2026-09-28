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

export interface JevStateEntry {
	/** Unique key for this state in the batch output. */
	id: string;
	/** The state to judge (string or object). */
	state: unknown;
}

export interface JevAdapterConfig {
	baseURL?: string;
	apiKey?: string;
	model?: string;
	maxTokens?: number;
	timeoutMs?: number;
	retries?: number;
	/**
	 * reasoning_effort for the decision model ('none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max').
	 * Hidden reasoning dominates wall-clock latency (measured: baseline ~36s/1100 tok vs 'low' ~1.2s/41 tok).
	 * NOTE: 'none' does NOT disable thinking on GLM-5.3 servers — it zeroes reasoning_tokens
	 * accounting but the thinking leaks into `content` as prose (unparseable). 'low' is the true fast switch.
	 */
	reasoningEffort?: string;
	/**
	 * response_format sent to the server. Default 'json_object': vLLM guarantees
	 * syntactically-valid JSON, which removes the parse-retry tier, with zero
	 * behavioral or latency cost (measured). Do NOT use 'json_schema' here:
	 * constrained decoding at temperature 0 collapses every answer to one-hot
	 * argmax (kills the probability-distribution paradigm; cannot express sum-to-1).
	 * 'none' sends no response_format (legacy behavior).
	 */
	responseFormat?: "json_object" | "none";
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
	/** Present when the model dropped one or more asked questions (unparseable
	 *  or omitted answers). Check this before trusting a partial answer set. */
	missingQuestions?: string[];
	usage?: { inputTokens: number; outputTokens: number };
	/** Wall-clock time of the decision request, ms. */
	elapsedMs?: number;
	/** Which attempt succeeded (1-based). >1 means a retry happened — check for slowness. */
	attempts?: number;
	/** Batch mode only: state ids the model failed to answer (omitted or empty). */
	missingStates?: string[];
	/** Present when a retry rescued the call: the unparseable first attempt. */
	debug?: { firstFailedRaw: string };
}

/** Multi-state batch result: answers[stateId][questionId]. */
export interface JevBatchResult {
	channel: string;
	answers: Record<string, JevDecisionResult["answers"]>;
	caveat: string;
	/** Batch mode only: question ids that went unanswered for at least one answered
	 *  state, mapped to the state ids missing them. A state that answered NOTHING
	 *  is reported via missingStates instead (not double-counted here). */
	missingQuestions?: Record<string, string[]>;
	usage?: { inputTokens: number; outputTokens: number };
	elapsedMs?: number;
	attempts?: number;
	missingStates?: string[];
	debug?: { firstFailedRaw: string };
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

/** Locate the decision model in ~/.pi/agent/models.json (or a path override). Returns null when unreadable or unmatched. Cached per key for the process lifetime — models.json is not expected to change mid-session. */
const endpointCache = new Map<string, EndpointInfo | null>();
export function loadEndpointFromModelsJson(path?: string): EndpointInfo | null {
	let file = path ?? process.env.JEV_MODELS_JSON;
	if (!file) {
		const home = process.env.HOME ?? process.env.USER_PROFILE ?? "";
		file = `${home}/.pi/agent/models.json`;
	}
	const key = `${file}:${process.env.JEV_BASE_URL ?? ""}:${process.env.JEV_MODEL ?? ""}:${process.env.JEV_API_KEY ? "k" : ""}`;
	if (endpointCache.has(key)) return endpointCache.get(key) ?? null;
	let resolved: EndpointInfo | null = null;
	try {
		const raw = JSON.parse(readFileSync(file)) as ModelsJson;
		const providers = raw.providers ?? {};
		for (const provider of Object.values(providers)) {
			for (const model of provider.models ?? []) {
				const id = model.id ?? "";
				if (id.includes(TARGET_MODEL_MATCH) && provider.baseUrl) {
					resolved = {
						baseURL: provider.baseUrl.replace(/\/$/, ""),
						apiKey: provider.apiKey ?? "",
						model: id,
					};
					break;
				}
			}
			if (resolved) break;
		}
	} catch {
		/* unreadable or malformed models.json — treat as unresolved */
	}
	endpointCache.set(key, resolved);
	return resolved;
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
			shape[id] = 0.0;
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
			shape[id] = probs;
		}
	}
	// FLAT output shape: {"qid":{"cand":p,...}} — no nested "probabilities" key.
	// Decode-bound latency: the model types every output token; the flat form
	// measured ~25% fewer completion tokens than the nested form on 10-question batches.
	lines.push(
		"",
		"── OUTPUT (this JSON only, flat: boolean → number, choice/score → {candidate: probability}) ──",
		JSON.stringify(shape),
	);
	return lines.join("\n");
}

/**
 * Build the batch decision prompt: N independent states judged against the
 * SAME questions in one request. Beats N parallel calls — the local endpoint
 * serializes on the GPU, and each request would otherwise pay its own
 * reasoning + queue overhead.
 */
export function buildPromptBatch(
	entries: JevStateEntry[],
	questions: Record<string, JevQuestion>,
): string {
	const lines = [
		"You are a decision scorer. You are given MULTIPLE independent states; judge each state against every question. Output a probability distribution for every (state, question) pair.",
		"Rules:",
		"1. Probabilities must reflect your genuine judgement. Never spread them uniformly out of laziness. The probabilities of each question must sum to 1.",
		"2. If you are genuinely uncertain, spread the probability across multiple candidates honestly — do NOT force a fake 1.0 onto one option. Your uncertainty is itself a useful signal: the caller routes on it (auto / ask-a-human / escalate), and a dishonest 1.0 breaks that mechanism. Only assign near-1 probabilities when the evidence is unambiguous.",
		"3. Output ONLY JSON. No explanations, no markdown fences.",
		"4. Use the given candidate names exactly as JSON keys, keyed first by state id, then by question id.",
	];
	for (const e of entries) {
		lines.push("", `── STATE ${e.id} ──`, typeof e.state === "string" ? e.state : JSON.stringify(e.state));
	}
	lines.push("", "── QUESTIONS (apply to EVERY state above) ──");
	const shape: Record<string, unknown> = {};
	let i = 0;
	for (const id of Object.keys(questions)) {
		i += 1;
		const q = questions[id];
		lines.push(`${i}. ${id} [${q.type}] ${q.instructions ?? ""}`);
		if (q.type === "boolean") {
			lines.push("   Output your estimate of P(true), between 0 and 1.");
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
		}
	}
	// FLAT nested shape: {stateId: {qid: {cand: p}}} — booleans are plain numbers.
	const qShape: Record<string, unknown> = {};
	for (const id of Object.keys(questions)) {
		const q = questions[id];
		if (q.type === "boolean") {
			qShape[id] = 0.0;
		} else {
			const probs: Record<string, number> = {};
			candidatesOf(q).forEach((c) => {
				probs[c.name] = 0.0;
			});
			qShape[id] = probs;
		}
	}
	for (const e of entries) shape[e.id] = qShape;
	lines.push(
		"",
		"── OUTPUT (this JSON only, flat, keyed by state id then question id) ──",
		JSON.stringify(shape),
	);
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
		reasoningEffort: opts.reasoningEffort ?? process.env.JEV_REASONING_EFFORT ?? "low",
		responseFormat:
			opts.responseFormat ??
			(process.env.JEV_RESPONSE_FORMAT as "json_object" | "none" | undefined) ??
			"json_object",
	};
}

async function callOpenAiCompatible(
	cfg: Required<JevAdapterConfig>,
	prompt: string,
	signal?: AbortSignal,
): Promise<{ raw: string; usage?: { inputTokens: number; outputTokens: number }; attempts: number }> {
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
					...(cfg.reasoningEffort ? { reasoning_effort: cfg.reasoningEffort } : {}),
					...(cfg.responseFormat && cfg.responseFormat !== "none"
						? { response_format: { type: cfg.responseFormat } }
						: {}),
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
			attempts: attempt + 1,
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
			| number
			| undefined;
		if (a === undefined) continue;
		if (q.type === "boolean") {
			// flat form: plain number or bool; string/JSON-number also coerced;
			// nested form: {probability} / {p}
			const p = Number(
				typeof a === "number" || typeof a === "boolean"
					? a
					: typeof a === "string"
						? a
						: (a.probability ?? a.p),
				);
			if (!Number.isFinite(p)) continue;
			const clamped = Math.max(0, Math.min(1, p));
			answers[id] = { type: "boolean", probability: Number(clamped.toFixed(4)) };
		} else {
			const cands = candidatesOf(q);
			if (!cands.length) continue;
			let probs = cands.map((c) => {
				// flat form: parsed[id] is {cand: p}; nested form: parsed[id].probabilities
				const obj = typeof a === "object" ? (a as Record<string, unknown>) : undefined;
				const flat = obj?.[c.name];
				const v = Number(flat !== undefined ? flat : obj?.probabilities?.[c.name]);
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
	const { prompt, parse } = runDecide(cfg, buildPrompt(state, questions), signal);
	const { usage, elapsedMs, attempts, parsed, firstFailedRaw } = await parse();
	const answers = assembleAnswers(questions, parsed);
	if (!Object.keys(answers).length) {
		throw new Error("Model output JSON matched none of the question IDs");
	}
	const missingQuestions = Object.keys(questions).filter((qid) => !(qid in answers));
	return {
		channel: `local decision model (${cfg.model})`,
		answers,
		caveat:
			"Probabilities are the model's self-reported estimates, not mathematically calibrated. Review high-risk decisions manually.",
		usage,
		elapsedMs,
		attempts,
		...(missingQuestions.length ? { missingQuestions } : {}),
		...(firstFailedRaw ? { debug: { firstFailedRaw } } : {}),
	};
}

/**
 * Run ONE batch decision request over multiple independent states (same
 * questions applied to each). One request beats N parallel calls: the local
 * endpoint serializes on the GPU and each request pays its own reasoning +
 * queue overhead. Output decode still scales with states×questions, so keep
 * both counts lean.
 */
export async function decideBatch(
	opts: JevAdapterConfig,
	entries: JevStateEntry[],
	questions: Record<string, JevQuestion>,
	signal?: AbortSignal,
): Promise<JevBatchResult> {
	if (!entries.length) throw new Error("decideBatch: entries must be non-empty");
	const ids = new Set<string>();
	for (const e of entries) {
		if (!e.id || ids.has(e.id)) throw new Error(`decideBatch: state ids must be unique and non-empty (got: "${e.id}")`);
		ids.add(e.id);
	}
	const cfg = resolveConfig(opts);
	if (!cfg.model) {
		throw new Error("jev-adapter: no decision model configured (set JEV_MODEL or add one to ~/.pi/agent/models.json).");
	}
	const { prompt, parse } = runDecide(cfg, buildPromptBatch(entries, questions), signal);
	const { usage, elapsedMs, attempts, parsed, firstFailedRaw } = await parse();
	const answers: JevBatchResult["answers"] = {};
	const missingStates: string[] = [];
	// question id -> state ids that answered the batch but dropped this question
	const missingQuestions = new Map<string, string[]>();
	for (const e of entries) {
		const perState = parsed[e.id] as Record<string, unknown> | undefined;
		if (!perState) {
			missingStates.push(e.id);
			continue;
		}
		answers[e.id] = assembleAnswers(questions, perState);
		if (!Object.keys(answers[e.id]).length) {
			delete answers[e.id];
			missingStates.push(e.id);
			continue;
		}
		// Partial miss: the state answered the batch but silently dropped some
		// question(s). Surface it — a silently-missing score once deranked a whole
		// review ranking with no warning anywhere.
		for (const qid of Object.keys(questions)) {
			if (!(qid in answers[e.id])) {
				const sids = missingQuestions.get(qid) ?? [];
				sids.push(e.id);
				missingQuestions.set(qid, sids);
			}
		}
	}
	if (!Object.keys(answers).length) {
		throw new Error("Model output JSON matched none of the state IDs");
	}
	return {
		channel: `local decision model (${cfg.model})`,
		answers,
		caveat:
			"Probabilities are the model's self-reported estimates, not mathematically calibrated. Review high-risk decisions manually.",
		usage,
		elapsedMs,
		attempts,
		...(missingStates.length ? { missingStates } : {}),
		...(missingQuestions.size
			? { missingQuestions: Object.fromEntries(missingQuestions) }
			: {}),
		...(firstFailedRaw ? { debug: { firstFailedRaw } } : {}),
	};
}

/** Build the prompt, then return a parse() closure implementing the retry-on-mangled-JSON loop. */
function runDecide(
	cfg: Required<JevAdapterConfig>,
	prompt: string,
	signal?: AbortSignal,
): {
	prompt: string;
	parse: () => Promise<{
		usage: JevDecisionResult["usage"];
		elapsedMs: number;
		attempts: number;
		parsed: Record<string, unknown>;
		/** Present when a retry happened: the unparseable first-attempt raw (for post-mortems). */
		firstFailedRaw?: string;
	}>;
} {
	return {
		prompt,
		parse: async () => {
			const t0 = Date.now();
			// A mangled JSON output wastes the whole request; retrying immediately (no
			// backoff) is far cheaper than the agent-level round trip of re-calling the tool.
			let raw = "";
			let usage: JevDecisionResult["usage"];
			let attempts = 0;
			let parsed: Record<string, unknown> | null = null;
			let firstFailedRaw: string | undefined;
			while (!parsed && attempts < cfg.retries) {
				attempts += 1; // attempts = number of requests actually made
				const r = await callOpenAiCompatible(cfg, prompt, signal);
				raw = r.raw;
				// usage accumulates across attempts so the token accounting stays honest
				usage = usage
					? {
							inputTokens: usage.inputTokens + (r.usage?.inputTokens ?? 0),
							outputTokens: usage.outputTokens + (r.usage?.outputTokens ?? 0),
						}
					: r.usage;
				parsed = extractJson(raw);
				if (!parsed && attempts === 1) firstFailedRaw = String(raw).slice(0, 400);
			}
			const elapsedMs = Date.now() - t0;
			if (!parsed) {
				throw new Error(
					`Could not parse JSON from model output after ${attempts} attempt(s) (last 200 chars: ${String(raw).slice(-200)})`,
				);
			}
			return { usage, elapsedMs, attempts, parsed, firstFailedRaw };
		},
	};
}
