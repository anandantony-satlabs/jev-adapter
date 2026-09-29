// Unit test: the decision model is CONFIG, not code.
// Exercises jev-adapter.config.json end to end with a mocked fetch and throwaway
// models.json / config files — no endpoint, no network. Covers: preference order,
// provider isolation, effort clamping, the 400 self-heal, and config validation.
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const dir = mkdtempSync(join(tmpdir(), "jev-cfg-test-"));
const write = (name, obj) => {
	const p = join(dir, name);
	writeFileSync(p, JSON.stringify(obj, null, 2));
	return p;
};

// --- captured requests -------------------------------------------------
const requests = [];
let respond = () => ({});
let failNextWithEffort400 = false;

globalThis.fetch = async (url, opts) => {
	const body = JSON.parse(String(opts.body));
	requests.push({ url: String(url), body });
	if (failNextWithEffort400) {
		failNextWithEffort400 = false;
		return {
			ok: false,
			status: 400,
			text: async () =>
				`{"error":{"message":"Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.","type":"BadRequestError"}}`,
		};
	}
	return {
		ok: true,
		status: 200,
		json: async () => ({
			choices: [{ message: { role: "assistant", content: JSON.stringify(respond(body, requests.length)) } }],
			usage: { prompt_tokens: 10, completion_tokens: 10 },
		}),
	};
};

// --- env isolation: everything comes from our temp files ---------------
for (const k of ["JEV_BASE_URL", "JEV_MODEL", "JEV_API_KEY", "JEV_REASONING_EFFORT", "JEV_MAX_TOKENS", "JEV_TIMEOUT_MS", "JEV_RETRIES", "JEV_RESPONSE_FORMAT"]) {
	delete process.env[k];
}
const modelsJson = write("models.json", {
	providers: {
		"provider-a": {
			baseUrl: "http://a.example/v1/",
			apiKey: "key-a",
			models: [{ id: "vendor/local-inference-lab/Qwen3.8-Flash-Next-NVFP4" }],
		},
		"provider-b": {
			baseUrl: "http://b.example/v1",
			apiKey: "key-b",
			models: [{ id: "vendor/local-inference-lab/GLM-5.3-Flash-NVFP4" }],
		},
	},
});
process.env.JEV_MODELS_JSON = modelsJson;

const { resolveConfig, decide, loadAdapterConfig, clampEffort, describeJevConfig } = await import("../../adapter.ts");

let failures = 0;
const check = (name, cond, extra = "") => {
	console.log(`${cond ? "PASS" : "FAIL"} ${name}${cond ? "" : `  ${extra}`}`);
	if (!cond) failures += 1;
};

const Q = { risk: { type: "choice", instructions: "risk", criteria: { docs: "docs only", rtl: "semantics change" } } };
const cfgFor = (name, obj) => write(name, obj);

// --- 1. config preference order decides which model runs --------------
const preferQwen = cfgFor(
	"prefer-qwen.json",
	{
		defaults: { reasoningEffort: "low", maxTokens: 2048 },
		decisionModels: [
			{ match: "Qwen3.8-Flash-Next-NVFP4", reasoningEffort: "none" },
			{ match: "GLM-5.3-Flash", reasoningEffort: "low" },
		],
	},
);
process.env.JEV_CONFIG = preferQwen;
let r = resolveConfig();
check("config: first entry in decisionModels wins", r.model.includes("Qwen"), r.model);
check("config: baseURL comes from the matching provider", r.baseURL === "http://a.example/v1", r.baseURL);
check("config: apiKey comes from that provider", r.apiKey === "key-a", r.apiKey);
check("config: per-entry effort beats file defaults", r.reasoningEffort === "none" && r.effortApplied === "none", r.reasoningEffort);
check("config: defaults.maxTokens applied", r.maxTokens === 2048, String(r.maxTokens));
check("config: no validation problems", r.configProblems.length === 0, JSON.stringify(r.configProblems));

// reorder the SAME file content → the other model runs (proves nothing is hard-coded)
const preferGlm = cfgFor(
	"prefer-glm.json",
	{
		defaults: { reasoningEffort: "low" },
		decisionModels: [
			{ match: "GLM-5.3-Flash", reasoningEffort: "low" },
			{ match: "Qwen3.8-Flash-Next-NVFP4", reasoningEffort: "none" },
		],
	},
);
process.env.JEV_CONFIG = preferGlm;
r = resolveConfig();
check("config: reordering decisionModels switches the model", r.model.includes("GLM"), r.model);
check("config: provider follows the switched model", r.baseURL === "http://b.example/v1" && r.apiKey === "key-b", `${r.baseURL} ${r.apiKey}`);

// --- 2. no match → actionable error naming what was wanted and what exists
const noMatch = cfgFor("no-match.json", { decisionModels: [{ match: "SomeOtherModel-7B" }] });
process.env.JEV_CONFIG = noMatch;
let err = "";
try {
	resolveConfig();
} catch (e) {
	err = e.message;
}
check(
	"config: unmatched model gives an actionable error",
	/SomeOtherModel-7B/.test(err) && /Qwen3\.8-Flash-Next-NVFP4/.test(err),
	err,
);

// --- 3. JEV_MODEL must not borrow another provider's URL/key -----------
process.env.JEV_CONFIG = preferQwen;
process.env.JEV_MODEL = "vendor/local-inference-lab/GLM-5.3-Flash-NVFP4";
r = resolveConfig();
check(
	"env: JEV_MODEL uses ITS OWN provider, not the preferred one",
	r.baseURL === "http://b.example/v1" && r.apiKey === "key-b",
	`${r.baseURL} ${r.apiKey}`,
);
delete process.env.JEV_MODEL;

// --- 4. effort clamping ------------------------------------------------
const clampCfg = cfgFor("clamp.json", {
	decisionModels: [
		{ match: "Qwen3.8-Flash-Next-NVFP4", supportedEfforts: ["none", "low"], effortMap: { high: "low" } },
	],
});
process.env.JEV_CONFIG = clampCfg;
check("clamp: mapped effort is remapped", clampEffort("high", loadAdapterConfig().decisionModels[0]).applied === "low");
const dropped = clampEffort("max", loadAdapterConfig().decisionModels[0]);
check("clamp: unmappable effort is dropped, not sent", dropped.applied === "" && dropped.dropped === true, JSON.stringify(dropped));

respond = () => ({ risk: { docs: 0.8, rtl: 0.2 } });
requests.length = 0;
let out = await decide({ reasoningEffort: "high" }, { commit: "x" }, Q);
check("clamp: request carried the MAPPED effort", requests.at(-1)?.body.reasoning_effort === "low", JSON.stringify(requests.at(-1)?.body));
check("clamp: result reports the clamp", out.effortClampedFrom === "high" && out.reasoningEffort === "low", JSON.stringify({ e: out.effortClampedFrom, a: out.reasoningEffort }));
check("result carries the model that answered", String(out.model).includes("Qwen"), String(out.model));

requests.length = 0;
out = await decide({ reasoningEffort: "max" }, { commit: "x" }, Q);
check("clamp: unmappable effort omits the field entirely", !("reasoning_effort" in requests.at(-1).body), JSON.stringify(requests.at(-1)?.body));
check("clamp: unmappable effort is reported as dropped", out.effortDropped === "max" && out.reasoningEffort === "", JSON.stringify({ d: out.effortDropped, e: out.reasoningEffort }));

// --- 5. unknown endpoint that 400s on our effort self-heals ------------
const unconfigured = cfgFor("unconfigured.json", { decisionModels: [{ match: "Qwen3.8-Flash-Next-NVFP4" }] });
process.env.JEV_CONFIG = unconfigured;
failNextWithEffort400 = true;
requests.length = 0;
out = await decide({ reasoningEffort: "ultra" }, { commit: "x" }, Q);
check("400 self-heal: decision still returned", out.answers.risk?.choice === "docs", JSON.stringify(out.answers));
check("400 self-heal: two requests, second without the field", requests.length === 2 && !("reasoning_effort" in requests[0].body) === false && !("reasoning_effort" in requests[1].body), JSON.stringify(requests.map((q) => q.body.reasoning_effort)));
check("400 self-heal: reported as dropped", out.effortDropped === "ultra", JSON.stringify({ d: out.effortDropped }));

// --- 6. config validation surfaces typos --------------------------------
const typo = cfgFor("typo.json", {
	decisionModels: [{ match: "Qwen3.8-Flash-Next-NVFP4", maxTokns: 999, reasoningEffort: "turbo", supportedEfforts: ["low"] }],
});
process.env.JEV_CONFIG = typo;
const rep = describeJevConfig();
check("validate: unknown tuning key reported", rep.problems.some((p) => /maxTokns/.test(p)), JSON.stringify(rep.problems));
check("validate: default effort outside supportedEfforts reported", rep.problems.some((p) => /turbo/.test(p)), JSON.stringify(rep.problems));
check("validate: entry still usable (not dropped)", rep.candidates.length === 1 && rep.candidates[0].match.includes("Qwen"), JSON.stringify(rep.candidates));
check("validate: report lists served models", rep.served.length === 2, JSON.stringify(rep.served));

// --- 7. an explicit JEV_CONFIG that is unreadable must NOT silently fall back
process.env.JEV_CONFIG = join(dir, "does-not-exist.json");
let hardErr = "";
try {
	resolveConfig();
} catch (e) {
	hardErr = e.message;
}
check("JEV_CONFIG missing file is a hard error", /unreadable|JEV_CONFIG/.test(hardErr), hardErr);

rmSync(dir, { recursive: true, force: true });
console.log(failures ? `\n${failures} FAILURE(S)` : "\nall config checks passed");
process.exit(failures ? 1 : 0);
