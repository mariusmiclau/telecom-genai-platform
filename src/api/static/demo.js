const form = document.querySelector("#query-form");
const queryInput = document.querySelector("#query");
const submitButton = document.querySelector("#submit-button");
const answerState = document.querySelector("#answer-state");
const answerContent = document.querySelector("#answer-content");
const answerText = document.querySelector("#answer-text");
const sourceList = document.querySelector("#source-list");
const confidence = document.querySelector("#confidence");
const traceStatus = document.querySelector("#trace-status");
const traceList = document.querySelector(".trace-list");
const traceLink = document.querySelector("#trace-link");
const evidenceList = document.querySelector("#evidence-list");
const claimList = document.querySelector("#claim-list");
const safetyNotices = document.querySelector("#safety-notices");
const conflictList = document.querySelector("#conflict-list");
let currentAnswerId = null;

function renderInvestigation(data) {
  currentAnswerId = data.answer_id || null;
  document.querySelector("#feedback-panel").hidden = !currentAnswerId;
  document.querySelector("#feedback-status").textContent = "";
  document.querySelector("#replay-result").textContent = "";
  document.querySelector("#feedback-comment").value = "";
  document.querySelector("#feedback-correction").value = "";
  document.querySelector("#feedback-citations").value = "";
  const needed = data.clarification?.required_fields || [];
  document.querySelector("#clarification-notice").textContent = needed.length ? `Clarification needed: ${needed.join(", ")}. Complete the scope fields above and run the same question again.` : "";
  const container = document.querySelector("#troubleshooting-sections");
  container.replaceChildren();
  document.querySelector("#troubleshooting-panel").hidden = !data.troubleshooting;
  if (data.troubleshooting) {
    for (const [key, title] of [["symptoms", "Symptoms (reported, unverified)"], ["relevant_incidents", "Relevant historical incidents"], ["likely_causes", "Historical cause hypotheses — not current diagnosis"], ["suggested_read_only_checks", "Suggested read-only checks (not executed)"], ["missing_evidence", "Missing evidence"]]) {
      const heading = document.createElement("h4"); heading.textContent = title;
      const list = document.createElement("ul");
      (data.troubleshooting[key] || []).forEach((fact) => {
        const row = document.createElement("li"); row.textContent = typeof fact === "string" ? fact : fact.text;
        (fact.citation_ids || []).forEach((id) => row.append(evidenceLink(id, `[${id}]`)));
        list.append(row);
      });
      if (!list.children.length) { const row = document.createElement("li"); row.textContent = "No source-supported item found."; list.append(row); }
      container.append(heading, list);
    }
  }
  const usage = data.consumption || {};
  document.querySelector("#consumption-summary").textContent = `Reserved upper bound: ${usage.reserved_token_upper_bound || 0}/${usage.maximum_request_tokens || 0} tokens; provider-reported completion usage: ${usage.provider_reported_tokens || 0}; completion attempts: ${usage.completion_attempts || 0}. No currency cost measured.`;
}

document.querySelector("#feedback-submit").addEventListener("click", async () => {
  if (!currentAnswerId) return;
  const answerId = currentAnswerId;
  try {
    const response = await fetch(`/api/v1/answers/${encodeURIComponent(answerId)}/feedback`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({
      rating: document.querySelector("#feedback-rating").value,
      incorrect_citation_ids: document.querySelector("#feedback-citations").value.split(",").map((value) => value.trim()).filter(Boolean),
      comment: document.querySelector("#feedback-comment").value, proposed_correction: document.querySelector("#feedback-correction").value,
    })});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Feedback rejected");
    if (currentAnswerId === answerId) document.querySelector("#feedback-status").textContent = `Feedback ${data.id}: ${data.state}. A different verified reviewer must approve corrections in live mode.`;
  } catch (error) { if (currentAnswerId === answerId) document.querySelector("#feedback-status").textContent = String(error.message); }
});

for (const [button, suffix, method] of [["#record-button", "", "GET"], ["#replay-button", "/replay", "POST"]]) {
  document.querySelector(button).addEventListener("click", async () => {
    if (!currentAnswerId) return;
    const answerId = currentAnswerId;
    try {
      const response = await fetch(`/api/v1/answers/${encodeURIComponent(answerId)}${suffix}`, {method});
      const data = await response.json();
      if (currentAnswerId === answerId) document.querySelector("#replay-result").textContent = JSON.stringify(data, null, 2);
    } catch (error) { if (currentAnswerId === answerId) document.querySelector("#replay-result").textContent = String(error.message); }
  });
}

function evidenceLink(id, label) {
  const link = document.createElement("a");
  link.href = `#evidence-${encodeURIComponent(id)}`;
  link.textContent = label;
  link.addEventListener("click", () => {
    const card = document.getElementById(`evidence-${id}`);
    if (card) card.open = true;
  });
  return link;
}

function renderEvidence(data) {
  evidenceList.replaceChildren();
  claimList.replaceChildren();
  safetyNotices.replaceChildren();
  conflictList.replaceChildren();
  const freshness = {historical: "Historical incident — not a current instruction", current: "Within review window (metadata)", unknown: "Freshness unknown"};
  const byDocument = new Map();
  (data.evidence || []).forEach((item) => {
    byDocument.set(item.document_id, item.citation_id);
    const card = document.createElement("details");
    card.id = `evidence-${item.citation_id}`;
    card.className = "evidence-card";
    const summary = document.createElement("summary");
    summary.textContent = `${item.citation_id} · ${item.document_id} · ${freshness[item.freshness] || "Freshness unknown"}`;
    const metadata = document.createElement("p");
    metadata.className = "helper";
    metadata.textContent = [item.document, item.redacted && "Excerpt safety-redacted", item.reviewed_at && `Reviewed: ${item.reviewed_at}`, item.valid_until && `Valid until: ${item.valid_until}`, item.software_version && `Software: ${item.software_version}`, item.equipment_family && `Equipment: ${item.equipment_family}`].filter(Boolean).join(" · ");
    const passage = document.createElement("pre");
    passage.textContent = item.passage;
    card.append(summary, metadata, passage);
    evidenceList.append(card);
  });
  (data.claims || []).forEach((claim) => {
    const item = document.createElement("li");
    const text = document.createElement("span");
    text.textContent = `${claim.claim} `;
    item.append(text);
    claim.citation_ids.forEach((id) => item.append(evidenceLink(id, `[${id}]`)));
    claimList.append(item);
  });
  const notices = new Map();
  (data.warnings || []).forEach((notice) => notices.set(notice.code + notice.message, notice.message));
  notices.forEach((message) => {
    const item = document.createElement("li");
    item.textContent = message;
    safetyNotices.append(item);
  });
  (data.conflicts || []).forEach((conflict) => {
    const item = document.createElement("li");
    const label = document.createElement("p");
    label.textContent = `${conflict.scope}: ${conflict.fact}`;
    item.append(label);
    conflict.values.forEach((value) => {
      const row = document.createElement("p");
      row.textContent = `${value.value} — `;
      value.source_ids.forEach((source) => {
        const id = byDocument.get(source);
        if (id) row.append(evidenceLink(id, source));
      });
      item.append(row);
    });
    conflictList.append(item);
  });
  document.querySelector("#claim-panel").hidden = !(data.claims?.length);
  document.querySelector("#conflict-panel").hidden = !(data.conflicts?.length);
  document.querySelector("#answer-kind").textContent = data.answer_kind === "historical_resolution" ? "Historical resolution · no live verification or action approval" : data.outcome === "escalated" ? "Human review required" : data.answer_kind === "advisory" ? "Source-based advisory · verify before acting" : "No answer published";
}

document.querySelectorAll("[data-query]").forEach((button) => {
  button.addEventListener("click", () => {
    queryInput.value = button.dataset.query;
    queryInput.focus();
  });
});

queryInput.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.key === "Enter") {
    form.requestSubmit();
  }
});

function setLoading() {
  currentAnswerId = null;
  submitButton.disabled = true;
  submitButton.textContent = "Running query";
  answerContent.hidden = true;
  confidence.hidden = true;
  traceList.classList.remove("is-complete");
  traceStatus.textContent = "Processing";
  answerState.hidden = false;
  answerState.className = "empty-state";
  answerState.innerHTML = '<div class="loading-lines" aria-label="Loading response"><span></span><span></span><span></span></div>';
}

function setError(message) {
  answerState.hidden = false;
  answerState.className = "empty-state error";
  answerState.textContent = message;
  answerContent.hidden = true;
  traceStatus.textContent = "Failed";
}

function sourceLabel(source) {
  if (typeof source === "string") return source;
  const section = source.section ? `, ${source.section}` : "";
  return `${source.document || "Unknown source"}${section}`;
}

function setSuccess(data) {
  answerState.hidden = true;
  answerContent.hidden = false;
  answerText.textContent = data.response;
  confidence.hidden = false;
  confidence.textContent = `${Math.round(data.confidence * 100)}% retrieval score`;
  sourceList.replaceChildren();
  renderEvidence(data);
  renderInvestigation(data);

  if (!data.sources?.length) {
    const item = document.createElement("li");
    item.textContent = "No source was returned.";
    sourceList.append(item);
  } else {
    data.sources.forEach((source) => {
      const item = document.createElement("li");
      item.textContent = sourceLabel(source);
      sourceList.append(item);
    });
  }

  traceStatus.textContent = data.outcome === "escalated" ? "Review required" : data.outcome === "abstained" ? "No eligible answer" : "Complete";
  traceList.classList.add("is-complete");
  if (data.trace_id) {
    traceLink.href = `http://localhost:16686/trace/${data.trace_id}`;
    traceLink.textContent = `Open trace ${data.trace_id.slice(0, 8)}`;
  } else {
    traceLink.href = "http://localhost:16686";
    traceLink.textContent = "Open Jaeger";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = queryInput.value.trim();
  if (!message) {
    setError("Enter an operations question before running the query.");
    return;
  }

  setLoading();
  try {
    const response = await fetch("/api/v1/demo/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, context: Object.fromEntries([["element_id", "#scope-node"], ["interface", "#scope-interface"], ["region", "#scope-region"], ["software_version", "#scope-version"], ["equipment_family", "#scope-equipment"]].map(([key, selector]) => [key, document.querySelector(selector).value.trim()]).filter(([, value]) => value)) }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "The request could not be completed.");
    }
    setSuccess(data);
  } catch (error) {
    setError(error.message || "The request could not be completed.");
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Run grounded query";
  }
});
