// Unit test: shared-endpoint detection.
//
// The harness usually sits on a different box from the GPU, so "is the decision
// endpoint local?" is the WRONG question. The question that matters is whether
// decisions decode on the same server that generates the AGENT's own tokens
// ($PI_MODEL via models.json) — because then JEV buys context window and not
// compute, elapsedMs includes queue time behind the agent's own requests, and
// /jev-use would load different weights into the running session's own server.
// Mocks nothing but the filesystem + env: no endpoint needed.
process.env.JEV_BASE_URL = "http://mock.local/v1";
process.env.JEV_MODEL = "mock-decision-model";
process.env.JEV_API_KEY = "mock";

import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const dir = mkdtempSync(join(tmpdir(), "jev-shared-"));
const modelsJson = (providers) => {
	const f = join(dir, `models-${Object.keys(providers).join("-")}.json`);
	writeFileSync(f, JSON.stringify({ providers }));
	process.env.JEV_MODELS_JSON = f;
	return f;
};

const { detectSharedAgentEndpoint } = await import("../../adapter.ts");

let failures = 0;
const check = (name, cond, got) => {
	console.log(`${cond ? "PASS" : "FAIL"} ${name}${cond || got === undefined ? "" : `  (got: ${got})`}`);
	if (!cond) failures += 1;
};

const GPU = { baseUrl: "http://192-168-1-57.tailae22aa.ts.net:8000/v1", apiKey: "x", models: [{ id: "lab/Qwen3.8-Flash-Next-NVFP4" }] };
const DESKTOP = { baseUrl: "http://10.0.0.9:11434/v1", apiKey: "x", models: [{ id: "lab/glm-5.3-flash" }] };

const savedPiModel = process.env.PI_MODEL;
const setPi = (v) => (v === undefined ? delete process.env.PI_MODEL : (process.env.PI_MODEL = v));

// --- 1. harness identity unknown -> stay quiet, do not guess ---
setPi(undefined);
modelsJson({ "local-llm": GPU });
check("no $PI_MODEL: null (nothing inferred)", detectSharedAgentEndpoint({ baseURL: GPU.baseUrl, model: "lab/Qwen3.8-Flash-Next-NVFP4" }) === null);

// --- 2. the real topology: one GPU box serves agent AND decisions ---
setPi("lab/Qwen3.8-Flash-Next-NVFP4");
modelsJson({ "local-llm": GPU });
const s = detectSharedAgentEndpoint({ baseURL: "http://192-168-1-57.tailae22aa.ts.net:8000/v1", model: "lab/Qwen3.8-Flash-Next-NVFP4" });
check("same origin: flagged shared", s?.shared === true, JSON.stringify(s && { shared: s.shared, sameWeights: s.sameWeights }));
check("same origin: also same weights", s?.sameWeights === true);
check("same origin: basis names $PI_MODEL", /\$PI_MODEL/.test(s?.basis ?? ""), s?.basis);
check("same origin: note says window not compute", /CONTEXT WINDOW, not compute/.test(s?.note ?? ""));
check("same origin: note flags the /jev-use hazard", /\/jev-use/.test(s?.note ?? ""));
check("same origin: origin compare tolerates trailing slash", detectSharedAgentEndpoint({ baseURL: "http://192-168-1-57.tailae22aa.ts.net:8000/v1/", model: "lab/Qwen3.8-Flash-Next-NVFP4" })?.shared === true);
check("same origin: default http port == explicit :8000 form differs only by port", detectSharedAgentEndpoint({ baseURL: "http://192-168-1-57.tailae22aa.ts.net:9000/v1", model: "lab/Qwen3.8-Flash-Next-NVFP4" })?.shared === false);

// --- 3. genuinely separate: agent on one box, decisions on another ---
modelsJson({ "local-llm": GPU, ollama: DESKTOP });
const sep = detectSharedAgentEndpoint({ baseURL: DESKTOP.baseUrl, model: "lab/glm-5.3-flash" });
check("separate boxes: NOT flagged", sep === null, JSON.stringify(sep));

// --- 4. same weights declared under two providers (two routes, one build) ---
const s2 = detectSharedAgentEndpoint({ baseURL: DESKTOP.baseUrl, model: "lab/Qwen3.8-Flash-Next-NVFP4" });
check("same id, different route: conservative sameWeights", s2?.sameWeights === true && s2?.shared === false, JSON.stringify(s2 && { shared: s2.shared, sameWeights: s2.sameWeights }));
check("same id, different route: basis explains the inference", /same weights/.test(s2?.basis ?? ""), s2?.basis);

// --- 5. models.json unreadable: still catch the same-id case, never invent an origin ---
writeFileSync(join(dir, "broken.json"), "{ not json");
process.env.JEV_MODELS_JSON = join(dir, "broken.json");
const s3 = detectSharedAgentEndpoint({ baseURL: GPU.baseUrl, model: "lab/Qwen3.8-Flash-Next-NVFP4" });
check("unreadable models.json: same-id case still detected", s3?.sameWeights === true && s3?.shared === false, JSON.stringify(s3 && { shared: s3.shared, sameWeights: s3.sameWeights }));
const s4 = detectSharedAgentEndpoint({ baseURL: GPU.baseUrl, model: "lab/some-other-model" });
check("unreadable models.json: no origin, no false alarm", s4 === null, JSON.stringify(s4));

// --- 6. agent on a different model of the SAME server: shared, different weights ---
modelsJson({ "local-llm": { ...GPU, models: [{ id: "lab/Qwen3.8-Flash-Next-NVFP4" }, { id: "lab/glm-5.3-flash" }] } });
setPi("lab/glm-5.3-flash");
const s5 = detectSharedAgentEndpoint({ baseURL: GPU.baseUrl, model: "lab/Qwen3.8-Flash-Next-NVFP4" });
check("same server, other weights: shared true / sameWeights false", s5?.shared === true && s5?.sameWeights === false, JSON.stringify(s5 && { shared: s5.shared, sameWeights: s5.sameWeights }));

if (savedPiModel === undefined) delete process.env.PI_MODEL;
else process.env.PI_MODEL = savedPiModel;
rmSync(dir, { recursive: true, force: true });
console.log(failures ? `\n${failures} FAILURE(S)` : "\nall shared-endpoint checks passed");
process.exit(failures ? 1 : 0);
