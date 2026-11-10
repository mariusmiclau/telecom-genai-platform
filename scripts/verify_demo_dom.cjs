// Unit-check the actual renderer with a fake DOM. This is NOT browser/visual QA.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const ids = new Map();
const tags = [];
class Element {
  constructor(tag = "div") { this.tag = tag; this.children = []; this.events = {}; this.hidden = false; this.classList = {add() {}, remove() {}}; tags.push(tag); }
  set id(value) { this._id = value; ids.set(value, this); }
  get id() { return this._id; }
  set innerHTML(value) { throw new Error("Dynamic HTML is not permitted in evidence rendering"); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = [...nodes]; }
  addEventListener(name, fn) { this.events[name] = fn; }
}
const elements = new Map();
const document = {
  querySelector(selector) { if (!elements.has(selector)) elements.set(selector, new Element()); return elements.get(selector); },
  querySelectorAll() { return []; },
  createElement(tag) { return new Element(tag); },
  getElementById(id) { return ids.get(id); },
};
const context = vm.createContext({document, console});
vm.runInContext(fs.readFileSync("src/api/static/demo.js", "utf8"), context);
context.fixture = {response: "A historical answer", confidence: .8, sources: [{document: "ticket.json"}], outcome: "answered", answer_kind: "historical_resolution",
  evidence: [{citation_id: "source-1", document_id: "ticket", document: "https://attacker.invalid/not-fetched", freshness: "historical", passage: '</pre><img src="x" onerror="canary()">'}],
  claims: [{claim: "A claim", citation_ids: ["source-1"]}], warnings: [{code: "historical_resolution", message: "Historical only"}], conflicts: []};
vm.runInContext("setSuccess(fixture)", context);
const card = ids.get("evidence-source-1");
assert(card && card.tag === "details");
assert.equal(card.children[2].textContent, context.fixture.evidence[0].passage);
assert(!tags.includes("img") && !tags.includes("script"));
assert(elements.get("#answer-kind").textContent.includes("no live verification"));
const claimLink = elements.get("#claim-list").children[0].children[1];
assert.equal(claimLink.href, "#evidence-source-1");
claimLink.events.click();
assert.equal(card.open, true);
assert.equal(elements.get("#claim-panel").hidden, false);
context.fixture.conflicts = [{scope: "CORE-A", fact: "availability_target", values: [{value: "99.9%", source_ids: ["ticket"]}]}];
context.fixture.outcome = "escalated";
context.fixture.answer_kind = "conflict";
vm.runInContext("setSuccess(fixture)", context);
assert.equal(elements.get("#conflict-panel").hidden, false);
assert.equal(elements.get("#trace-status").textContent, "Review required");
context.fixture = {response: "No evidence", sources: [], confidence: 0, outcome: "abstained", answer_kind: "insufficient_evidence"};
vm.runInContext("setSuccess(fixture)", context);
assert.equal(elements.get("#evidence-list").children.length, 0);
assert.equal(elements.get("#claim-panel").hidden, true);
assert.equal(elements.get("#conflict-panel").hidden, true);
console.log("Demo renderer unit checks passed (fake DOM; no visual/browser verification)");
