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

  traceStatus.textContent = data.outcome === "abstained" ? "No evidence" : "Complete";
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
      body: JSON.stringify({ message }),
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
