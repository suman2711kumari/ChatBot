"use strict";
(() => {
  const form = document.getElementById("ragForm");
  const queryInput = document.getElementById("ragQuery");
  const runButton = document.getElementById("ragRun");
  const status = document.getElementById("ragStatus");
  const results = document.getElementById("ragResults");
  const compareButton = document.getElementById("ragCompare");
  const compareStatus = document.getElementById("ragCompareStatus");
  const compareResults = document.getElementById("ragCompareResults");

  async function request(endpoint, query) {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({query})
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(
      typeof payload.detail === "string" ? payload.detail : "Request failed"
    );
    return payload;
  }

  function renderSources(matches) {
    results.replaceChildren();
    if (!matches.length) {
      const message = document.createElement("p");
      message.className = "usage-empty";
      message.textContent = "Nothing matched. Try stress, studying, loneliness, sleep or breathing.";
      results.appendChild(message);
      return;
    }
    matches.forEach((item, i) => {
      const card = document.createElement("article");
      card.className = "rag-match";
      const heading = document.createElement("div");
      heading.className = "rag-match-top";
      const title = document.createElement("strong");
      title.textContent = (i + 1) + ". " + item.title;
      const score = document.createElement("span");
      score.className = "rag-similarity";
      score.textContent = "Similarity " + Number(item.similarity).toFixed(3);
      heading.append(title, score);
      const caption = document.createElement("p");
      caption.className = "rag-subheading";
      caption.textContent = "Topic: " + item.topic + " · Retriever: " + item.retrieval_method;
      const summary = document.createElement("p");
      summary.className = "rag-summary";
      summary.textContent = item.summary;
      card.append(heading, caption, summary);
      try {
        const url = new URL(item.url);
        if (url.protocol === "https:") {
          const link = document.createElement("a");
          link.href = url.href;
          link.textContent = "View original resource";
          link.className = "rag-source-link";
          link.rel = "noopener noreferrer";
          link.target = "_blank";
          card.appendChild(link);
        }
      } catch (_) {}
      results.appendChild(card);
    });
  }

  form.addEventListener("submit", async e => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;
    runButton.disabled = true;
    status.textContent = "Searching the local reference knowledge base…";
    results.replaceChildren();
    compareResults.replaceChildren();
    compareStatus.textContent = "";
    try {
      const data = await request("/admin/rag-preview", query);
      renderSources(data.matches || []);
      status.textContent = data.match_count +
        " relevant passage(s) retrieved. Gemini was not called. This preview does not save conversations.";
    } catch (err) {
      status.textContent = "Retrieval unavailable: " + err.message;
    } finally {
      runButton.disabled = false;
    }
  });

  compareButton.addEventListener("click", async () => {
    const query = queryInput.value.trim();
    if (!query || compareButton.disabled) return;
    compareButton.disabled = true;
    compareStatus.textContent = "Calling Gemini twice; results may take a moment…";
    compareResults.replaceChildren();
    try {
      const data = await request("/admin/rag-compare", query);
      const pairs = [
        ["Gemini without RAG", data.answer_without_rag],
        ["Gemini with retrieved context", data.answer_with_rag]
      ];
      pairs.forEach(([heading, answer]) => {
        const card = document.createElement("article");
        card.className = "rag-compare-card";
        const h = document.createElement("h3");
        h.textContent = heading;
        const p = document.createElement("p");
        p.textContent = answer;
        card.append(h, p);
        compareResults.appendChild(card);
      });
      compareStatus.textContent = "Two Gemini calls completed with " +
        data.retrieved_count + " retrieved notes. Similarity or difference is not an accuracy score.";
    } catch (err) {
      compareStatus.textContent = "Comparison unavailable: " + err.message;
    } finally {
      compareButton.disabled = false;
    }
  });
})();
