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
 *    (both local endpoints this runs against are reasoning models);
 *  - normalises probabilities that do not sum to 1;
 *  - retries with backoff on 429 / 5xx.
 *
 * NOTHING ABOUT THE MODEL IS HARD-CODED HERE. Which model is the decision
 * model, and how it is talked to (reasoning_effort vocabulary, token caps),
 * comes from jev-adapter.config.json next to this file — see `loadAdapterConfig`
 * for the search order. Switching endpoints is a JSON edit, not a code edit.
 */

import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// SECURITY NOTE: there is deliberately NO hard-coded endpoint fallback.
// The endpoint always comes from ~/.pi/agent/models.json (or JEV_BASE_URL /
// JEV_MODEL / JEV_API_KEY env overrides), and the *model choice* from
// jev-adapter.config.json. Do not commit real hostnames, tailnet names, or
// keys into this file or that one — both are meant for a public repo.

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
	 * Whether it is sendable at all, and what it buys, is ENDPOINT-SPECIFIC and lives in
	 * jev-adapter.config.json (`supportedEfforts` / `effortMap` / per-entry default) — the
	 * measured numbers are in that file's `note` fields. A value the server would reject is
	 * clamped, never sent as-is, and the clamp is reported back to the caller.
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
	/** Decision model id that answered (config-driven; check it when results look off). */
	model?: string;
	/** reasoning_effort actually sent ('' = omitted → server default). */
	reasoningEffort?: string;
	/** Set when the requested effort was not sendable and got clamped. */
	effortClampedFrom?: string;
	/** Set when the preferred decision model was unavailable and a later one answered. */
	modelFallbackFrom?: string;
	/** Set when the server rejected the requested effort outright (sent without the field). */
	effortDropped?: string;
	/** Set when an attempt was cut off by `max_tokens` (finish_reason="length") and
	 *  the retry ran with a larger output budget. A truncation is the most common
	 *  cause of "Could not parse JSON": the answer is simply incomplete. */
	maxTokensRaisedFrom?: number;
	/** Pre-flight output-budget estimate for this call (see estimateOutputBudget). */
	budget?: JevBudgetEstimate;
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
	model?: string;
	reasoningEffort?: string;
	effortClampedFrom?: string;
	effortDropped?: string;
	modelFallbackFrom?: string;
	maxTokensRaisedFrom?: number;
	budget?: JevBudgetEstimate;
}

/* ------------------------------------------------------------------ */
/* Adapter config: WHICH model decides, and how to talk to it          */
/*                                                                     */
/* Everything model-specific lives in jev-adapter.config.json so that  */
/* pointing the tool at a different decision model is a JSON edit.     */
/* ------------------------------------------------------------------ */

/** Tunables that can be set per model entry, in `defaults`, or per call. */
export interface JevTuning {
	reasoningEffort?: string;
	maxTokens?: number;
	timeoutMs?: number;
	retries?: number;
	responseFormat?: "json_object" | "none";
}

/** One `decisionModels` entry: a model selector plus its endpoint quirks. */
export interface JevModelProfile extends JevTuning {
	/** Exact model id, or a substring of one, as found in models.json. */
	match: string;
	/**
	 * Efforts this server accepts. A requested effort outside this list is NEVER
	 * sent as-is (vLLM answers those with HTTP 400 and the whole call is lost).
	 */
	supportedEfforts?: string[];
	/** Requested → sendable, consulted before giving up on a clamped effort. */
	effortMap?: Record<string, string>;
	/** Free text: measured behaviour of this endpoint, surfaced by /jev-config. */
	note?: string;
	/**
	 * One or two sentences telling the CALLING model how to pick an effort on this
	 * endpoint. Kept short on purpose: it is rendered into the jev_decide tool
	 * description, i.e. it costs prompt tokens in every session. `note` is the
	 * long-form version, shown only by /jev-config.
	 */
	effortGuide?: string;
}

export interface AdapterConfigFile {
	/** Where it was read from (null = built-in defaults only). */
	path: string | null;
	source: "file" | "builtin";
	defaults: JevTuning;
	decisionModels: JevModelProfile[];
	/**
	 * Active preference (from $JEV_USE_MODEL and the local override file), in the
	 * order applied to `decisionModels`. Empty = plain file order. This is what
	 * makes switching between two endpoints a one-command thing.
	 */
	preferred: string[];
	/** Local preference file, when one exists. */
	localPath: string | null;
	/** Config problems found while validating — surfaced, never silently ignored. */
	problems: string[];
}

/**
 * Local (un-versioned) preference override — written by `/jev-use`, so switching
 * endpoints does not dirty the tracked config file. Shape: `{ "prefer": ["GLM"] }`.
 * `$JEV_LOCAL` overrides the path; `$JEV_USE_MODEL` is the same thing for one
 * process/session without touching any file (it wins over the file).
 */
export function adapterLocalPath(): string {
	return expandHome(process.env.JEV_LOCAL ?? "~/.pi/agent/" + LOCAL_FILENAME);
}

function loadPreferenceOverride(): { prefer: string[]; path: string | null; problems: string[] } {
	const file = adapterLocalPath();
	const problems: string[] = [];
	if (!existsSync(file)) return { prefer: [], path: null, problems };
	try {
		const raw = JSON.parse(readFileSync(file, "utf8")) as Record<string, unknown>;
		const prefer = Array.isArray(raw.prefer) ? raw.prefer.filter((p): p is string => typeof p === "string" && !!p) : [];
		if (!Array.isArray(raw.prefer) && raw.prefer !== undefined) problems.push(`${file}: "prefer" must be an array of strings — ignored`);
		for (const key of Object.keys(raw)) {
			if (key !== "prefer" && !key.startsWith("$")) problems.push(`${file}: unknown key "${key}" — ignored (only "prefer" is honoured here)`);
		}
		return { prefer, path: file, problems };
	} catch (e) {
		problems.push(`${file}: invalid JSON (${(e as Error).message}) — preference override ignored`);
		return { prefer: [], path: file, problems };
	}
}
/** Move the preferred entries to the front of `decisionModels` (stable otherwise). */
function applyPreference(cfg: AdapterConfigFile, prefer: string[]): void {
	if (!prefer.length || cfg.decisionModels.length < 2) return;
	const chosen: JevModelProfile[] = [];
	for (const q of prefer) {
		const hits = cfg.decisionModels.filter((m) => !chosen.includes(m) && looseMatch(m.match, q));
		if (!hits.length) {
			cfg.problems.push(`preference "${q}" matches no decisionModels entry (have: ${cfg.decisionModels.map((m) => m.match).join(", ")}) — ignored`);
			continue;
		}
		chosen.push(...hits);
	}
	cfg.decisionModels = [...chosen, ...cfg.decisionModels.filter((m) => !chosen.includes(m))];
	cfg.preferred = chosen.map((m) => m.match);
}

/** Persist the preference (used by /jev-use). Returns the file written. */
export function writeModelPreference(preference: string): string {
	const file = adapterLocalPath();
	let existing: Record<string, unknown> = {};
	try {
		existing = JSON.parse(readFileSync(file, "utf8")) as Record<string, unknown>;
	} catch {
		/* new file */
	}
	const prev = Array.isArray(existing.prefer) ? (existing.prefer as unknown[]).filter((p): p is string => typeof p === "string") : [];
	existing.prefer = [preference, ...prev.filter((p) => !looseMatch(p, preference))];
	mkdirSync(dirname(file), { recursive: true });
	writeFileSync(file, JSON.stringify(existing, null, 2) + "\n");
	configCache.clear();
	endpointCache.clear();
	return file;
}

/** Live model list from the endpoint itself (`GET {baseURL}/models`). */
export async function listServedModels(
	cfg: ResolvedJevConfig,
	signal?: AbortSignal,
): Promise<{ ok: boolean; models: string[]; error?: string }> {
	try {
		const res = await fetch(`${cfg.baseURL}/models`, {
			headers: cfg.apiKey ? { Authorization: `Bearer ${cfg.apiKey}` } : {},
			signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(10000)]) : AbortSignal.timeout(10000),
		});
		if (!res.ok) return { ok: false, models: [], error: `HTTP ${res.status}` };
		const data = (await res.json()) as { data?: { id?: string }[] };
		return { ok: true, models: (data.data ?? []).map((m) => m.id ?? "").filter(Boolean) };
	} catch (e) {
		return { ok: false, models: [], error: (e as Error).message };
	}
}

/** Built-in last resort; `defaults` in the config file normally overrides these. */
const BUILTIN_TUNING: Required<JevTuning> = {
	reasoningEffort: "low",
	// 4000 was measured too small: a 10-state x 3-question batch of score
	// questions decodes ~7.5k completion tokens and came back truncated
	// ("Could not parse JSON" after every retry, no rescue possible). See
	// estimateOutputBudget() for the per-answer decode costs.
	maxTokens: 8000,
	timeoutMs: 180000,
	retries: 3,
	responseFormat: "json_object",
};

const CONFIG_FILENAME = "jev-adapter.config.json";
const LOCAL_FILENAME = "jev-adapter.local.json";

/** Case-insensitive, either-direction substring match: "glm" ↔ "GLM-5.3-Flash". */
function looseMatch(a: string, b: string): boolean {
	const x = a.toLowerCase();
	const y = b.toLowerCase();
	return x.includes(y) || y.includes(x);
}

/** Public matcher for config entry names ("glm", "qwen3.8", full ids). */
export function nameMatches(a: string, b: string): boolean {
	return looseMatch(a, b);
}

/** This module's directory, tolerant of jiti (CJS) and plain node (ESM) hosts. */
const EXT_DIR = (() => {
	try {
		return dirname(fileURLToPath(import.meta.url));
	} catch {
		return process.cwd();
	}
})();

function expandHome(p: string): string {
	if (!p.startsWith("~")) return p;
	const home = process.env.HOME ?? process.env.USER_PROFILE ?? "";
	return join(home, p.slice(1));
}

/** Config search order: $JEV_CONFIG, then the repo copy, then the user copy. */
export function adapterConfigPaths(): { paths: string[]; explicit: boolean } {
	if (process.env.JEV_CONFIG) return { paths: [expandHome(process.env.JEV_CONFIG)], explicit: true };
	return {
		paths: [join(EXT_DIR, CONFIG_FILENAME), expandHome("~/.pi/agent/jev-adapter.json")],
		explicit: false,
	};
}

function statKey(file: string): string {
	try {
		const s = statSync(file);
		return `${file}:${s.mtimeMs}:${s.size}`;
	} catch {
		return `${file}:missing`;
	}
}

const TUNING_KEYS = ["reasoningEffort", "maxTokens", "timeoutMs", "retries", "responseFormat"] as const;

/** Copy the known tunables out of a JSON object, recording anything malformed. */
function validateTuning(raw: unknown, where: string, problems: string[]): JevTuning {
	const out: JevTuning = {};
	if (raw === undefined || raw === null) return out;
	if (typeof raw !== "object" || Array.isArray(raw)) {
		problems.push(`${where}: not an object — ignored`);
		return out;
	}
	const obj = raw as Record<string, unknown>;
	for (const key of Object.keys(obj)) {
		if (!TUNING_KEYS.includes(key as (typeof TUNING_KEYS)[number])) {
			problems.push(`${where}: unknown key "${key}" — ignored (typo?)`);
			continue;
		}
		const v = obj[key];
		if (key === "responseFormat") {
			if (v === "json_object" || v === "none") out.responseFormat = v;
			else problems.push(`${where}.responseFormat: "${String(v)}" is not "json_object" or "none" — ignored`);
		} else if (key === "reasoningEffort") {
			if (typeof v === "string" && v) out.reasoningEffort = v;
			else problems.push(`${where}.reasoningEffort: must be a non-empty string — ignored`);
		} else {
			const n = Number(v);
			if (!Number.isFinite(n) || n <= 0) problems.push(`${where}.${key}: "${String(v)}" is not a positive number — ignored`);
			else (out as unknown as Record<string, number>)[key] = n;
		}
	}
	return out;
}

function validateProfiles(raw: unknown, where: string, problems: string[]): JevModelProfile[] {
	if (raw === undefined || raw === null) return [];
	if (!Array.isArray(raw)) {
		problems.push(`${where}: must be an array — ignored`);
		return [];
	}
	const out: JevModelProfile[] = [];
	raw.forEach((entry, i) => {
		const at = `${where}[${i}]`;
		if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
			problems.push(`${at}: not an object — dropped`);
			return;
		}
		const e = entry as Record<string, unknown>;
		if (typeof e.match !== "string" || !e.match.trim()) {
			problems.push(`${at}: missing "match" (the model id / substring) — dropped`);
			return;
		}
		const RESERVED = ["match", "supportedEfforts", "effortMap", "note", "effortGuide"];
		const tunables: Record<string, unknown> = {};
		for (const [k, v] of Object.entries(e)) if (!RESERVED.includes(k)) tunables[k] = v;
		const profile: JevModelProfile = { match: e.match.trim(), ...validateTuning(tunables, at, problems) };
		if (Array.isArray(e.supportedEfforts)) {
			profile.supportedEfforts = e.supportedEfforts.filter((s): s is string => typeof s === "string" && !!s);
		}
		if (e.effortMap && typeof e.effortMap === "object" && !Array.isArray(e.effortMap)) {
			const m: Record<string, string> = {};
			for (const [k, v] of Object.entries(e.effortMap as Record<string, unknown>)) {
				if (typeof v === "string" && v) m[k] = v;
				else problems.push(`${at}.effortMap.${k}: must be a string — ignored`);
			}
			profile.effortMap = m;
		}
		if (typeof e.note === "string") profile.note = e.note;
		if (typeof e.effortGuide === "string") profile.effortGuide = e.effortGuide;
		if (profile.supportedEfforts?.length && profile.reasoningEffort && !profile.supportedEfforts.includes(profile.reasoningEffort)) {
			problems.push(`${at}: default reasoningEffort "${profile.reasoningEffort}" is not in supportedEfforts — every call will be clamped`);
		}
		out.push(profile);
	});
	return out;
}

const configCache = new Map<string, AdapterConfigFile>();

/**
 * Read the adapter config. Search order: explicit `path` arg / $JEV_CONFIG,
 * then <extension dir>/jev-adapter.config.json, then ~/.pi/agent/jev-adapter.json.
 * The first existing file wins (no merge). An explicit path that cannot be read
 * is a hard error — silently deciding with the wrong model is worse.
 */
export function loadAdapterConfig(path?: string): AdapterConfigFile {
	const { paths, explicit } = adapterConfigPaths();
	const wanted = path ? [expandHome(path)] : paths;
	const localFile = adapterLocalPath();
	const key = `${wanted.map(statKey).join("|")}:${explicit ? "x" : "s"}:${statKey(localFile)}:${process.env.JEV_USE_MODEL ?? ""}`;
	const cached = configCache.get(key);
	if (cached) return cached;

	const problems: string[] = [];
	let chosen: string | null = null;
	let parsed: Record<string, unknown> = {};
	for (const file of wanted) {
		if (!existsSync(file)) continue;
		let raw: string;
		try {
			raw = readFileSync(file, "utf8");
		} catch (e) {
			const msg = `${file}: unreadable (${(e as Error).message})`;
			if (explicit) throw new Error(`jev-adapter: JEV_CONFIG points at an unreadable file — ${msg}`);
			problems.push(msg + " — skipped");
			continue;
		}
		try {
			parsed = JSON.parse(raw) as Record<string, unknown>;
		} catch (e) {
			const msg = `${file}: invalid JSON (${(e as Error).message})`;
			if (explicit) throw new Error(`jev-adapter: JEV_CONFIG is not valid JSON — ${msg}`);
			problems.push(msg + " — skipped");
			continue;
		}
		chosen = file;
		break;
	}
	if (explicit && !chosen) {
		throw new Error(
			`jev-adapter: JEV_CONFIG points at ${wanted[0]}, which does not exist — refusing to decide with the built-in/repo config (the wrong decision model is worse than no decision).`,
		);
	}

	const cfg: AdapterConfigFile = {
		path: chosen,
		source: chosen ? "file" : "builtin",
		defaults: validateTuning(parsed.defaults, "defaults", problems),
		decisionModels: validateProfiles(parsed.decisionModels, "decisionModels", problems),
		preferred: [],
		localPath: null,
		problems,
	};
	if (chosen && !cfg.decisionModels.length) {
		problems.push(`${chosen}: no usable decisionModels entries — the model must come from JEV_MODEL / a models.json lookup by name`);
	}
	// Switching between two endpoints: $JEV_USE_MODEL (this process) then the local
	// override file (/jev-use) both reorder decisionModels without editing the file.
	const local = loadPreferenceOverride();
	problems.push(...local.problems);
	cfg.localPath = local.path;
	applyPreference(cfg, [...(process.env.JEV_USE_MODEL ? [process.env.JEV_USE_MODEL] : []), ...local.prefer]);
	configCache.set(key, cfg);
	return cfg;
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
	/** provider key in models.json — diagnostics only */
	provider: string;
}

function modelsJsonPath(path?: string): string {
	if (path) return expandHome(path);
	if (process.env.JEV_MODELS_JSON) return expandHome(process.env.JEV_MODELS_JSON);
	const home = process.env.HOME ?? process.env.USER_PROFILE ?? "";
	return join(home, ".pi", "agent", "models.json");
}

/** Every usable (baseUrl, model) pair in models.json. [] when unreadable/malformed. */
export function listModelEndpoints(path?: string): EndpointInfo[] {
	const file = modelsJsonPath(path);
	const out: EndpointInfo[] = [];
	try {
		const raw = JSON.parse(readFileSync(file, "utf8")) as ModelsJson;
		for (const [provider, p] of Object.entries(raw.providers ?? {})) {
			if (!p?.baseUrl) continue;
			for (const model of p.models ?? []) {
				const id = model?.id ?? "";
				if (!id) continue;
				out.push({ baseURL: p.baseUrl.replace(/\/$/, ""), apiKey: p.apiKey ?? "", model: id, provider });
			}
		}
	} catch {
		/* unreadable or malformed models.json — treated as "nothing available" */
	}
	return out;
}

/** exact id match beats substring; first profile in config order wins. */
function findByMatch(endpoints: EndpointInfo[], match: string): EndpointInfo | undefined {
	return (
		endpoints.find((e) => e.model === match) ?? endpoints.find((e) => e.model.includes(match))
	);
}

/** The profile whose `match` selects `modelId` (exact first, then substring). */
export function pickProfile(cfg: AdapterConfigFile, modelId: string): JevModelProfile | undefined {
	if (!modelId) return undefined;
	return (
		cfg.decisionModels.find((p) => p.match === modelId) ??
		cfg.decisionModels.find((p) => modelId.includes(p.match))
	);
}

/** Walk `decisionModels` in config order and return the first model actually served. */
export function pickPreferredEndpoint(
	endpoints: EndpointInfo[],
	cfg: AdapterConfigFile = loadAdapterConfig(),
): EndpointInfo | null {
	for (const profile of cfg.decisionModels) {
		const hit = findByMatch(endpoints, profile.match);
		if (hit) return hit;
	}
	return null;
}

/**
 * Locate the decision model in ~/.pi/agent/models.json (or a path override),
 * choosing between the models actually served using the config file's
 * preference order. Returns null when unreadable or unmatched. Cached per
 * (models.json mtime, config mtime, env) so config edits are picked up.
 */
const endpointCache = new Map<string, EndpointInfo | null>();
export function loadEndpointFromModelsJson(path?: string, cfg?: AdapterConfigFile): EndpointInfo | null {
	const file = modelsJsonPath(path);
	const conf = cfg ?? loadAdapterConfig();
	const key = `${statKey(file)}|${conf.path ?? "builtin"}|${conf.decisionModels.map((p) => p.match).join(",")}|${process.env.JEV_BASE_URL ?? ""}|${process.env.JEV_MODEL ?? ""}|${process.env.JEV_API_KEY ? "k" : ""}`;
	if (endpointCache.has(key)) return endpointCache.get(key) ?? null;
	const resolved = pickPreferredEndpoint(listModelEndpoints(file), conf);
	endpointCache.set(key, resolved ?? null);
	return resolved ?? null;
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

/** Remap a requested reasoning_effort to something the target server accepts. */
export function clampEffort(
	requested: string,
	profile: JevModelProfile | undefined,
): { applied: string; requested?: string; dropped?: boolean } {
	const supported = profile?.supportedEfforts;
	if (!requested) return { applied: "" };
	if (!supported?.length || supported.includes(requested)) return { applied: requested };
	const mapped = profile?.effortMap?.[requested];
	if (mapped && supported.includes(mapped)) return { applied: mapped, requested };
	// No sendable translation: omit the field (server default) rather than burn the
	// call on a guaranteed HTTP 400 — but tell the caller we did it.
	return { applied: "", requested, dropped: true };
}

/** Everything resolveConfig settled on, including the diagnostics /jev-config shows. */
export type ResolvedJevConfig = Required<JevAdapterConfig> & {
	/** jev-adapter.config.json actually used (null → built-in defaults only). */
	configPath: string | null;
	/** the config entry's `match` that selected this model. */
	matchedBy?: string;
	/** measured-behaviour note from that config entry. */
	profileNote?: string;
	/** reasoning_effort actually sent ('' → field omitted, server default). */
	effortApplied: string;
	/** set when the requested effort was not sendable and had to be remapped. */
	effortRequested?: string;
	/** set when the requested effort had no sendable translation (field omitted). */
	effortDropped?: string;
	/** active model preference (empty = plain config file order). */
	preferred: string[];
	/**
	 * The top preference was not declared in models.json, so the next entry in order
	 * answered. Silent fallback across endpoints is dangerous here (the effort
	 * semantics differ per endpoint), so it is reported, not hidden.
	 */
	preferredNotServed?: string;
	/** config validation problems that mattered to this resolution. */
	configProblems: string[];
};

/**
 * Resolve the effective config.
 * Layer order per field: call opts → JEV_* env → matched config entry →
 * config `defaults` → built-in. Throws when no endpoint can be resolved.
 */
export function resolveConfig(opts: JevAdapterConfig = {}): ResolvedJevConfig {
	const cfg = loadAdapterConfig();
	const served = listModelEndpoints();
	const requestedModel = opts.model ?? process.env.JEV_MODEL ?? "";
	const preferred = pickPreferredEndpoint(served, cfg);
	const model = requestedModel || preferred?.model || "";
	// The provider (baseUrl + key) must belong to the model actually chosen: when
	// JEV_MODEL names a model served by a DIFFERENT provider than the preferred
	// one, borrowing the preferred provider's URL would silently send the state —
	// and that provider's key — to the wrong endpoint.
	const endpoint =
		(model ? findByMatch(served, model) : undefined) ?? (requestedModel ? undefined : preferred);
	const profile = pickProfile(cfg, model);
	const baseURL = (opts.baseURL ?? process.env.JEV_BASE_URL ?? endpoint?.baseURL ?? "").replace(/\/$/, "");
	if (!baseURL) {
		const wanted = cfg.decisionModels.map((p) => `"${p.match}"`).join(", ") || "(none configured)";
		const ids = served.map((e) => e.model);
		throw new Error(
			`jev-adapter: no decision endpoint. The config looks for ${wanted}, but ~/.pi/agent/models.json `
				+ `serves ${ids.length ? ids.join(", ") : "nothing readable"}. Add a matching model (with a baseUrl provider) there, `
				+ `or point jev-adapter.config.json at one, or set JEV_BASE_URL + JEV_MODEL (and JEV_API_KEY if the endpoint needs auth).`,
		);
	}
	// SECURITY: whatever baseURL resolves to, the resolved apiKey is sent to
	// it as a Bearer header. Only point JEV_BASE_URL at endpoints you trust —
	// a hostile endpoint receives both the key and the judged state payload.
	const numTune = (key: "maxTokens" | "timeoutMs" | "retries", env: string): number =>
		(opts[key] ?? numEnv(env) ?? profile?.[key] ?? cfg.defaults[key] ?? BUILTIN_TUNING[key]) as number;
	const effortRequested =
		opts.reasoningEffort ??
		process.env.JEV_REASONING_EFFORT ??
		profile?.reasoningEffort ??
		cfg.defaults.reasoningEffort ??
		BUILTIN_TUNING.reasoningEffort;
	const clamped = clampEffort(effortRequested, profile);
	// Loud fallback: the preferred endpoint was not found among the declared models,
	// so a later entry answered. Say so — the caller may be assuming this endpoint's
	// effort vocabulary and latency.
	const preferredNotServed =
		!requestedModel && cfg.preferred.length && !(profile && nameMatches(cfg.preferred[0], profile.match))
			? cfg.preferred[0]
			: undefined;
	return {
		baseURL,
		apiKey: opts.apiKey ?? process.env.JEV_API_KEY ?? endpoint?.apiKey ?? "",
		model,
		maxTokens: numTune("maxTokens", "JEV_MAX_TOKENS"),
		timeoutMs: numTune("timeoutMs", "JEV_TIMEOUT_MS"),
		retries: numTune("retries", "JEV_RETRIES"),
		reasoningEffort: clamped.applied,
		responseFormat:
			opts.responseFormat ??
			(process.env.JEV_RESPONSE_FORMAT as JevTuning["responseFormat"] | undefined) ??
			profile?.responseFormat ??
			cfg.defaults.responseFormat ??
			BUILTIN_TUNING.responseFormat,
		configPath: cfg.path,
		matchedBy: profile?.match,
		profileNote: profile?.note,
		preferred: cfg.preferred,
		...(preferredNotServed ? { preferredNotServed } : {}),
		effortApplied: clamped.applied,
		...(clamped.requested && clamped.requested !== clamped.applied
			? { effortRequested: clamped.requested, ...(clamped.dropped ? { effortDropped: clamped.requested } : {}) }
			: {}),
		configProblems: cfg.problems,
	};
}

/** Fields every result carries about WHICH model answered and at what effort. */
function effortFields(
	cfg: ResolvedJevConfig,
	/** set when the server itself rejected the effort (HTTP 400) and we retried without it */
	serverRejected?: string,
): Pick<JevDecisionResult, "model" | "reasoningEffort" | "effortClampedFrom" | "effortDropped" | "modelFallbackFrom"> {
	const dropped = cfg.effortDropped ?? serverRejected;
	return {
		model: cfg.model,
		// what actually went on the wire, not what we wanted
		reasoningEffort: serverRejected ? "" : cfg.effortApplied,
		...(cfg.effortRequested ? { effortClampedFrom: cfg.effortRequested } : {}),
		...(dropped ? { effortDropped: dropped } : {}),
		...(cfg.preferredNotServed ? { modelFallbackFrom: cfg.preferredNotServed } : {}),
	};
}

/** Everything /jev-config needs to explain the current resolution. */
export interface JevConfigReport {
	configPath: string | null;
	configSource: "file" | "builtin";
	searchOrder: string[];
	problems: string[];
	served: string[];
	candidates: { match: string; selected: boolean; note?: string; effortGuide?: string; reasoningEffort?: string; supportedEfforts?: string[] }[];
	resolved: ResolvedJevConfig | null;
	error?: string;
}

export function describeJevConfig(): JevConfigReport {
	const cfg = loadAdapterConfig();
	const served = listModelEndpoints().map((e) => e.model);
	let resolved: ResolvedJevConfig | null = null;
	let error: string | undefined;
	try {
		resolved = resolveConfig();
	} catch (e) {
		error = (e as Error).message;
	}
	return {
		configPath: cfg.path,
		configSource: cfg.source,
		searchOrder: adapterConfigPaths().paths,
		problems: cfg.problems,
		served,
		candidates: cfg.decisionModels.map((p) => ({
			match: p.match,
			selected: !!resolved?.matchedBy && resolved.matchedBy === p.match,
			note: p.note,
			effortGuide: p.effortGuide,
			reasoningEffort: p.reasoningEffort,
			supportedEfforts: p.supportedEfforts,
		})),
		resolved,
		error,
	};
}

/* ------------------------------------------------------------------ */
/* Output budget                                                       */
/* ------------------------------------------------------------------ */

/**
 * Measured decode cost per answered question (thinking effort, temp 0) — what
 * a `max_tokens` budget has to cover. A score answer emits a whole level
 * distribution, a boolean answer one number:
 *   10 states x [score, score, boolean]  measured 7545 out  (~250/answer)
 *    6 states x [score, score, boolean]  measured 3380 out  (~190/answer)
 *    9 states x [choice, boolean]        measured 2077 out  (~115/answer)
 * Long state digests push score answers up, so these skew pessimistic for
 * boolean-heavy calls. Prefill (the state text itself) is ~100x cheaper and is
 * deliberately NOT counted here.
 */
const OUTPUT_TOKENS_PER_ANSWER: Record<string, number> = {
	score: 280,
	choice: 160,
	boolean: 130,
};

/** Ceiling for the automatic max_tokens raise (see runDecide). */
const MAX_TOKENS_CEILING = 32000;

export interface JevBudgetEstimate {
	/** states x questions — the decode surface. */
	answers: number;
	estOutputTokens: number;
	maxTokens: number;
	/** 1 = expected to fit in one response; >1 = split the batch into this many calls. */
	chunks: number;
}

/**
 * Pre-flight check that a call's OUTPUT can fit in one response.
 *
 * Decode is the latency bottleneck (~60 tok/s against ~5000 tok/s prefill) and a
 * request that overruns `max_tokens` does not fail gracefully: the JSON comes
 * back cut off mid-object, extractJson fails, and every retry fails the same way
 * because the shape never changed — the whole batch is lost and the caller has
 * to re-derive its digests. Saying so before spending the round trips is the
 * cheapest fix available; runDecide additionally auto-raises on truncation.
 */
export function estimateOutputBudget(
	stateCount: number,
	questions: Record<string, JevQuestion>,
	maxTokens: number,
): JevBudgetEstimate {
	const qids = Object.keys(questions ?? {});
	const perState = qids.reduce(
		(sum, qid) => sum + (OUTPUT_TOKENS_PER_ANSWER[questions[qid]?.type] ?? 200),
		0,
	);
	const states = Math.max(1, stateCount);
	const estOutputTokens = states * (perState || 200);
	return {
		answers: states * Math.max(1, qids.length),
		estOutputTokens,
		maxTokens,
		chunks: Math.max(1, Math.ceil(estOutputTokens / Math.max(1, maxTokens))),
	};
}

async function callOpenAiCompatible(
	cfg: ResolvedJevConfig,
	prompt: string,
	signal?: AbortSignal,
	/** per-attempt override, raised when a previous attempt was cut off */
	maxTokens?: number,
): Promise<{
	raw: string;
	usage?: { inputTokens: number; outputTokens: number };
	attempts: number;
	/** set when the server rejected our reasoning_effort (400) and we retried without it */
	effortRejected?: string;
	/** set when the response hit the output limit (finish_reason="length") */
	truncated?: boolean;
}> {
	let lastErr = "";
	// Local, mutable copy: an endpoint whose effort vocabulary we don't know
	// (no config entry, or a stale supportedEfforts list) answers with HTTP 400.
	let sendEffort = cfg.reasoningEffort;
	let effortRejected: string | undefined;
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
					max_tokens: maxTokens ?? cfg.maxTokens,
					temperature: 0,
					...(sendEffort ? { reasoning_effort: sendEffort } : {}),
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
			const body = (await res.text()).slice(0, 250);
			// Self-heal for an endpoint we have no config entry for: a 400 about
			// reasoning effort means its effort vocabulary differs from what we sent.
			// Drop the field (the 400 cost ~30ms) and keep the decision — but surface
			// it, because a silent downgrade would hide a real config gap.
			if (res.status === 400 && sendEffort && /reasoning[_ ]?effort/i.test(body)) {
				effortRejected = sendEffort;
				sendEffort = "";
				lastErr = `HTTP 400: reasoning_effort "${effortRejected}" rejected, retrying without it`;
				continue;
			}
			// Switching endpoints leaves a window where the box serves the OTHER model
			// (one GPU, one vLLM process). "model does not exist" is that case, and an
			// opaque 404 dump wastes a whole debugging round — ask the endpoint what it
			// does serve and say so.
			if (
				(res.status === 400 || res.status === 404) &&
				/model/i.test(body) &&
				/does not exist|not found|unknown model|invalid model/i.test(body)
			) {
				const served = await listServedModels(cfg);
				lastErr =
					`HTTP ${res.status}: the endpoint at ${cfg.baseURL} does not serve "${cfg.model}"` +
					(served.ok
						? ` — it currently serves ${served.models.length ? served.models.join(", ") : "(nothing)"}`
						: ` (live /models probe failed: ${served.error})`) +
					`. Is the other decision server up? Switch the model with /jev-use <name> (current preference lives in ${adapterLocalPath()}).`;
				break;
			}
			lastErr = `HTTP ${res.status}: ${body}`;
			break;
		}
		const data = (await res.json()) as {
			choices?: {
				message?: { content?: string; reasoning_content?: string };
				finish_reason?: string;
			}[];
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
			...(data.choices?.[0]?.finish_reason === "length" ? { truncated: true } : {}),
			...(effortRejected ? { effortRejected } : {}),
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
				const obj = typeof a === "object" && a !== null ? (a as Record<string, unknown>) : undefined;
				const nested = obj?.probabilities as Record<string, unknown> | undefined;
				const flat = obj?.[c.name];
				const v = Number(flat !== undefined ? flat : nested?.[c.name]);
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
		throw new Error(
			"jev-adapter: no decision model configured — add one to decisionModels in jev-adapter.config.json (and make sure ~/.pi/agent/models.json serves it), or set JEV_MODEL.",
		);
	}
	const { prompt, parse } = runDecide(cfg, buildPrompt(state, questions), signal);
	const { usage, elapsedMs, attempts, parsed, firstFailedRaw, effortRejected, maxTokensRaisedFrom } = await parse();
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
		...effortFields(cfg, effortRejected),
		...(maxTokensRaisedFrom ? { maxTokensRaisedFrom } : {}),
		budget: estimateOutputBudget(1, questions, cfg.maxTokens),
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
		throw new Error(
			"jev-adapter: no decision model configured — add one to decisionModels in jev-adapter.config.json (and make sure ~/.pi/agent/models.json serves it), or set JEV_MODEL.",
		);
	}
	const { prompt, parse } = runDecide(cfg, buildPromptBatch(entries, questions), signal);
	const { usage, elapsedMs, attempts, parsed, firstFailedRaw, effortRejected, maxTokensRaisedFrom } = await parse();
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
		...effortFields(cfg, effortRejected),
		...(maxTokensRaisedFrom ? { maxTokensRaisedFrom } : {}),
		budget: estimateOutputBudget(entries.length, questions, cfg.maxTokens),
		...(missingStates.length ? { missingStates } : {}),
		...(missingQuestions.size
			? { missingQuestions: Object.fromEntries(missingQuestions) }
			: {}),
		...(firstFailedRaw ? { debug: { firstFailedRaw } } : {}),
	};
}

/** Build the prompt, then return a parse() closure implementing the retry-on-mangled-JSON loop. */
function runDecide(
	cfg: ResolvedJevConfig,
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
		/** Present when the server rejected our reasoning_effort (400) and we retried without it. */
		effortRejected?: string;
		/** Present when an attempt was cut off by max_tokens and the retry got a bigger budget. */
		maxTokensRaisedFrom?: number;
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
			let effortRejected: string | undefined;
			let maxTokens = cfg.maxTokens;
			let maxTokensRaisedFrom: number | undefined;
			while (!parsed && attempts < cfg.retries) {
				attempts += 1; // attempts = number of requests actually made
				const r = await callOpenAiCompatible(cfg, prompt, signal, maxTokens);
				if (r.effortRejected) effortRejected = r.effortRejected;
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
				// Truncation is the one failure a blind retry cannot rescue: the same shape
				// overruns the same budget every time and the whole batch is lost. Raise the
				// output budget for the next attempt instead of asking again.
				if (!parsed && r.truncated && maxTokens < MAX_TOKENS_CEILING) {
					maxTokensRaisedFrom ??= maxTokens;
					maxTokens = Math.min(maxTokens * 2, MAX_TOKENS_CEILING);
				}
			}
			const elapsedMs = Date.now() - t0;
			if (!parsed) {
				throw new Error(
					`Could not parse JSON from model output after ${attempts} attempt(s)` +
						(maxTokensRaisedFrom
							? ` - the answer kept being cut off by the output limit (last tried: ${maxTokens} tokens), so this batch is too big for one response: split it into smaller calls (fewer states, or fewer score questions per call)`
							: "") +
						` (last 200 chars: ${String(raw).slice(-200)})`,
				);
			}
			return {
				usage,
				elapsedMs,
				attempts,
				parsed,
				firstFailedRaw,
				...(effortRejected ? { effortRejected } : {}),
				...(maxTokensRaisedFrom ? { maxTokensRaisedFrom } : {}),
			};
		},
	};
}
