// Unit test: silently-dropped questions must be reported, not vanish.
// Mocks globalThis.fetch so no endpoint is needed; exercises decide() and
// decideBatch() partial-answer accounting (the bug this patch fixes: a score
// question was silently omitted from a 15-state batch with zero warnings).
process.env.JEV_BASE_URL = "http://mock.local/v1";
process.env.JEV_MODEL = "mock-decision-model";
process.env.JEV_API_KEY = "mock";

// OpenAI-compatible mock: returns whatever the fixture says for the given prompt marker.
const scenarios = new Map();
globalThis.fetch = async (_url, opts) => {
	const body = JSON.parse(String(opts.body));
	const marker = body.messages.at(-1).content;
	const hit = [...scenarios.keys()].find((k) => marker.includes(k));
	if (!hit) throw new Error(`no scenario for prompt: ${marker.slice(0, 120)}`);
	const content = scenarios.get(hit);
	return {
		ok: true,
		status: 200,
		json: async () => ({
			choices: [{ message: { role: "assistant", content: JSON.stringify(content) } }],
			usage: { prompt_tokens: 10, completion_tokens: 10 },
		}),
	};
};

const { decide, decideBatch } = await import("../../adapter.ts");

const questions = {
	risk: { type: "choice", instructions: "risk class", criteria: ["docs", "mixed", "test", "rtl"] },
	need_review: { type: "boolean", instructions: "needs review?" },
};

let failures = 0;
const check = (name, cond) => {
	console.log(`${cond ? "PASS" : "FAIL"} ${name}`);
	if (!cond) failures += 1;
};

// --- scenario 1: batch, one state drops one question (partial answer) ---
scenarios.set("STATE-A-FULL-STATE-B-PARTIAL", {
	"stateA": { risk: { docs: 0.9, mixed: 0.1 }, need_review: 0.05 },
	"stateB": { risk: { docs: 0.5, mixed: 0.5 } }, // need_review silently dropped
});
const batch = await decideBatch({}, [
	{ id: "stateA", state: "STATE-A-FULL-STATE-B-PARTIAL alpha" },
	{ id: "stateB", state: "STATE-A-FULL-STATE-B-PARTIAL beta" },
], questions);
check("batch: stateA has both answers", "risk" in batch.answers.stateA && "need_review" in batch.answers.stateA);
check("batch: stateB keeps its partial answers", "risk" in batch.answers.stateB && !("need_review" in batch.answers.stateB));
check("batch: missingQuestions recorded", JSON.stringify(batch.missingQuestions) === JSON.stringify({ need_review: ["stateB"] }));

// --- scenario 2: batch, one state drops EVERYTHING (must be missingStates, not missingQuestions) ---
scenarios.set("STATE-C-GONE", {
	"stateA": { risk: { docs: 0.8, mixed: 0.2 }, need_review: 0.1 },
	// stateC absent entirely
});
const batch2 = await decideBatch({}, [
	{ id: "stateA", state: "STATE-C-GONE alpha" },
	{ id: "stateC", state: "STATE-C-GONE gamma" },
], questions);
check("batch2: fully-missing state -> missingStates", JSON.stringify(batch2.missingStates) === JSON.stringify(["stateC"]));
check("batch2: no spurious missingQuestions", batch2.missingQuestions === undefined);

// --- scenario 3: single decide, one question dropped ---
scenarios.set("SINGLE-PARTIAL", { risk: { test: 0.6, rtl: 0.4 } }); // need_review dropped
const single = await decide({}, "SINGLE-PARTIAL some state", questions);
check("single: answered question present", "risk" in single.answers);
check("single: missingQuestions lists dropped id", JSON.stringify(single.missingQuestions) === JSON.stringify(["need_review"]));

// --- scenario 4: single decide, all questions answered -> no warning field ---
scenarios.set("SINGLE-FULL", { risk: { test: 0.6, rtl: 0.4 }, need_review: 0.9 });
const full = await decide({}, "SINGLE-FULL clean state", questions);
check("single full: no missingQuestions field", full.missingQuestions === undefined);

process.exit(failures ? 1 : 0);
