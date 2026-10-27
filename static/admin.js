"use strict";
(() => {
  const refresh = document.getElementById("refreshButton");
  const status = document.getElementById("refreshStatus");
  const active = document.getElementById("activeUsers");
  const count = document.getElementById("messageCount");
  const average = document.getElementById("averageSentiment");
  const averageCaption = document.getElementById("averageSentimentCaption");
  const sentimentSelect = document.getElementById("sentimentModel");
  const sentimentDataStatus = document.getElementById("sentimentDataStatus");
  let chosenSentimentModel = null;
  const usage = document.getElementById("modelUsage");
  const empty = document.getElementById("chartEmpty");
  const canvas = document.getElementById("sentimentChart");
  let chart;

  function modelTitle(model) {
    const names = {
      gemini: "Gemini API",
      dialogpt_fallback: "Local DialoGPT",
      safety_message: "Safety message",
      unavailable: "Unavailable"
    };
    return names[model] || String(model).replaceAll("_", " ");
  }
  function drawUsage(models) {
    usage.replaceChildren();
    const entries = Object.entries(models || {}).filter(([, n]) => Number.isFinite(n) && n > 0);
    if (entries.length === 0) {
      const note = document.createElement("p");
      note.className = "usage-empty";
      note.textContent = "No generated responses yet. Once you start chatting, model activity will appear here.";
      usage.appendChild(note);
      return;
    }
    const total = entries.reduce((sum, [, n]) => sum + n, 0);
    for (const [name, number] of entries.sort((a, b) => b[1] - a[1])) {
      const row = document.createElement("div");
      row.className = "usage-row";
      const head = document.createElement("div");
      head.className = "usage-row-head";
      const title = document.createElement("strong");
      title.textContent = modelTitle(name);
      const value = document.createElement("span");
      value.textContent = String(number) + " · " + Math.round(number / total * 100) + "%";
      head.append(title, value);
      const track = document.createElement("div");
      track.className = "usage-track";
      const fill = document.createElement("div");
      fill.className = "usage-fill";
      fill.style.width = (100 * number / total) + "%";
      track.appendChild(fill);
      row.append(head, track);
      usage.appendChild(row);
    }
  }

  function drawChart(trend, selectedModel) {
    const dates = Array.isArray(trend?.dates) ? trend.dates : [];
    const values = Array.isArray(trend?.avg_scores) ? trend.avg_scores : [];
    const hasPoints = dates.length > 0 && dates.length === values.length;
    empty.hidden = hasPoints;
    if (chart) { chart.destroy(); chart = null; }
    if (!hasPoints) return;
    if (typeof Chart === "undefined") {
      empty.hidden = false;
      empty.textContent = "Chart library could not load. The numeric metrics still work.";
      return;
    }
    const labels = dates.map(date => {
      // Database UTC timestamps have 'YYYY-MM-DD HH:MM' format.
      const utcDate = new Date(String(date).replace(" ", "T") + ":00Z");
      if (Number.isNaN(utcDate.getTime())) return String(date);
      return utcDate.toLocaleString([], {hour: "numeric", minute: "2-digit", month: "short", day: "numeric"});
    });
    chart = new Chart(canvas, {
      type: "line",
      data: {
        labels,
        datasets: [{
          label: "Average " + (selectedModel === "roberta" ? "RoBERTa" : "VADER") + " sentiment",
          data: values,
          borderColor: "#6650aa",
          backgroundColor: "rgba(102,80,170,.09)",
          fill: true,
          tension: .35,
          pointRadius: 4,
          pointBackgroundColor: "#6650aa",
          borderWidth: 2.5
        }]
      },
      options: {
        maintainAspectRatio: false,
        responsive: true,
        interaction: {mode: "index", intersect: false},
        plugins: {legend: {display: false}, tooltip: {callbacks: {
          label: context => "Average " + (selectedModel === "roberta" ? "RoBERTa" : "VADER") + ": " + Number(context.raw).toFixed(3)
        }}},
        scales: {
          y: {
            min: -1, max: 1,
            grid: {color: "#f0edf6"},
            ticks: {stepSize: .5, color: "#8d879a"}
          },
          x: {grid: {display: false}, ticks: {maxTicksLimit: 6, color: "#8d879a"}}
        }
      }
    });
  }

  async function loadStats() {
    refresh.disabled = true;
    refresh.textContent = "Refreshing…";
    try {
      const url = chosenSentimentModel
        ? "/admin/stats?sentiment_model=" + encodeURIComponent(chosenSentimentModel)
        : "/admin/stats";
      const response = await fetch(url, {cache: "no-store"});
      if (!response.ok) throw Error("Server returned " + response.status);
      const data = await response.json();
      chosenSentimentModel = data.sentiment_model || "vader";
      sentimentSelect.value = chosenSentimentModel;
      averageCaption.textContent = (chosenSentimentModel === "roberta"
        ? "RoBERTa positive − negative probability"
        : "VADER compound score") + " · −1 to +1";
      sentimentDataStatus.textContent = (data.sentiment_message_count ?? 0)
        + " message(s) scored by " + (chosenSentimentModel === "roberta" ? "RoBERTa" : "VADER")
        + " in the last 24 hours. Response usage counts include all messages.";
      active.textContent = String(data.total_active_users ?? 0);
      count.textContent = String(data.message_count ?? 0);
      average.textContent = typeof data.average_sentiment === "number"
        ? (data.average_sentiment >= 0 ? "+" : "") + data.average_sentiment.toFixed(2)
        : "—";
      drawChart(data.sentiment_trend, chosenSentimentModel);
      drawUsage(data.model_usage);
      status.textContent = "Last refreshed " + new Date().toLocaleTimeString([], {hour: "numeric", minute: "2-digit"});
    } catch (error) {
      status.textContent = "Refresh failed: " + error.message;
      active.textContent = count.textContent = average.textContent = "—";
      empty.hidden = false;
      empty.textContent = "Could not load analytics. Please check your server connection.";
    } finally {
      refresh.disabled = false;
      refresh.textContent = "↻   Refresh data";
    }
  }
  sentimentSelect.addEventListener("change", () => {
    chosenSentimentModel = sentimentSelect.value;
    loadStats();
  });
  refresh.addEventListener("click", loadStats);
  loadStats();
  setInterval(loadStats, 5 * 60 * 1000);
})();
