// Unit test: the OUTPUT budget. Two failure modes this guards:
//  1. estimateOutputBudget() must predict, before the call, whether the answer
//     set fits in one response (a too-big batch is the #1 cause of a lost call).
//  2. A response cut off by max_tokens (finish_reason="length") must NOT be
//     retried blindly with the same shape — that fails identically every time.
//     runDecide has to raise max_tokens for the next attempt instead.
// Mocks globalThis.fetch, so no endpoint is needed.
process.env.JEV_BASE_URL = "http://mock.local/v1";
process.env.JEV_MODEL = "mock-decision-model";
process.env.JEV_API_KEY = "mock";
// Deliberately small, so the truncation path is reachable in a test.
process.env.JEV_MAX_TOKENS = "4000";

/** One scenario = the answers object plus what the mock pretends happened. */
const scenarios = new Map();
/** Bodies of every request the adapter actually sent, in order. */
const sentBodies = [];

globalThis.fetch = async (_url, opts) => {
	const body = JSON.parse(String(opts.body));
	sentBodies.push(body);
	const marker = body.messages.at(-1).content;
	const hit = [...scenarios.keys()].find((k) => marker.includes(k));
	if (!hit) throw new Error(`no scenario for prompt: ${marker.slice(0, 120)}`);
	const sc = scenarios.get(hit);
	// Simulate "this answer needs more room": truncate only while the requested
	// budget is below what the scenario says the answer really needs.
	const trunc = sc.truncated && body.max_tokens < (sc.truncateBelow ?? Infinity);
	return {
		ok: true,
		status: 200,
		json: async () => ({
			choices: [
				{
					message: { role: "assistant", content: trunc ? sc.truncatedContent : JSON.stringify(sc.answers) },
					...(trunc ? { finish_reason: "length" } : {}),
				},
			],
			usage: { prompt_tokens: 10, completion_tokens: 10 },
		}),
	};
};

const { decideBatch, estimateOutputBudget } = await import("../../adapter.ts");

let failures = 0;
const check = (name, cond, got) => {
	console.log(`${cond ? "PASS" : "FAIL"} ${name}${cond || got === undefined ? "" : `  (got: ${got})`}`);
	if (!cond) failures += 1;
};

const questions = {
	// two score questions + a boolean = the exact shape that lost the real call
	priority: { type: "score", instructions: "review priority", criteria: ["skip", "low", "medium", "high", "critical"] },
	risk: { type: "score", instructions: "correctness risk", criteria: ["none", "low", "medium", "high", "severe"] },
	reobservable: { type: "boolean", instructions: "is the evidence re-runnable" },
};

/* --- 1. the estimator: ~280/160/130 output tokens per answered question --- */
const e10 = estimateOutputBudget(10, questions, 8000);
check("est: 10x[score,score,bool] answers=30", e10.answers === 30, String(e10.answers));
check("est: 10x[score,score,bool] tokens=6900", e10.estOutputTokens === 6900, String(e10.estOutputTokens));
check("est: fits at 8000 (chunks=1)", e10.chunks === 1, String(e10.chunks));
const e10small = estimateOutputBudget(10, questions, 4000);
check("est: same batch needs 2 chunks at 4000", e10small.chunks === 2, String(e10small.chunks));
const eBools = estimateOutputBudget(9, { a: { type: "boolean", instructions: "x" }, b: { type: "choice", instructions: "x", criteria: ["p", "q"] } }, 4000);
check("est: boolean+choice costs less than two scores", eBools.estOutputTokens === 9 * (130 + 160), String(eBools.estOutputTokens));
const eSingle = estimateOutputBudget(1, questions, 8000);
check("est: single-state mode still budgeted", eSingle.answers === 3 && eSingle.chunks === 1, JSON.stringify(eSingle));

/* --- 2. truncation is rescued by RAISING the budget, not by re-asking --- */
scenarios.set("TRUNC-THEN-FIT", {
	truncated: true,
	truncateBelow: 8000, // the answer only fits once the budget is raised past 4000
	truncatedContent: '{"s1":{"priority":{"0":0.1,"1":0.2"', // cut off mid-object
	answers: { s1: { priority: { "3": 0.6, "4": 0.4 }, risk: { "3": 0.5, "4": 0.5 }, reobservable: 0.8 } },
});
sentBodies.length = 0;
const rescued = await decideBatch({}, [{ id: "s1", state: "TRUNC-THEN-FIT" }], questions);
check("trunc: the batch survived on the retry", Object.keys(rescued.answers).length === 1);
check("trunc: two requests were made", sentBodies.length === 2, String(sentBodies.length));
check("trunc: retry carried a LARGER max_tokens", sentBodies[1].max_tokens > sentBodies[0].max_tokens, `${sentBodies[0].max_tokens} -> ${sentBodies[1].max_tokens}`);
check("trunc: first request used the configured budget", sentBodies[0].max_tokens === 4000, String(sentBodies[0].max_tokens));
check("trunc: maxTokensRaisedFrom reported", rescued.maxTokensRaisedFrom === 4000, String(rescued.maxTokensRaisedFrom));
check("trunc: attempts reported as 2 (retried)", rescued.attempts === 2, String(rescued.attempts));
check("trunc: budget attached for calibration", rescued.budget?.estOutputTokens === 690, JSON.stringify(rescued.budget));

/* --- 3. a batch that never fits must fail with actionable advice, not silence --- */
scenarios.set("TRUNC-FOREVER", {
	truncated: true,
	truncatedContent: '{"s1":{"priority":{"0":0.1',
	answers: {},
});
let advice = "";
try {
	await decideBatch({}, [{ id: "s1", state: "TRUNC-FOREVER" }], questions);
} catch (e) {
	advice = String(e.message);
}
check("over-budget: throws", advice.length > 0);
check("over-budget: says the batch is too big / split it", /too big|split/i.test(advice), advice.slice(0, 160));
check("over-budget: does not claim a plain parse failure only", /output limit/.test(advice), advice.slice(0, 160));

process.exit(failures ? 1 : 0);
