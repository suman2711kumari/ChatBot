"use strict";
(() => {
  const form = document.getElementById("chatForm");
  const input = document.getElementById("userInput");
  const send = document.getElementById("sendBtn");
  const log = document.getElementById("chatMessages");
  const status = document.getElementById("chatStatus");
  const counter = document.getElementById("charCounter");
  const userId = (() => {
    try {
      let value = localStorage.getItem("mindspace_user_id") || localStorage.getItem("user_id");
      if (!value) {
        value = "user_" + (crypto.randomUUID?.() || Math.random().toString(36).slice(2));
        localStorage.setItem("mindspace_user_id", value);
      }
      return value;
    } catch (_) {
      return "session_" + Math.random().toString(36).slice(2);
    }
  })();

  function fitInput() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 130) + "px";
    counter.textContent = input.value.length + " / 2000";
  }
  input.addEventListener("input", fitInput);
  input.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      if (!send.disabled && input.value.trim()) form.requestSubmit();
    }
  });
  document.querySelectorAll("[data-prompt]").forEach(button => {
    button.addEventListener("click", () => {
      input.value = button.dataset.prompt;
      fitInput();
      input.focus();
    });
  });

  function addUserSentiment(messageElement, score, model) {
    if (typeof score !== "number" || !Number.isFinite(score)) return;
    const metadata = messageElement.querySelector(".bubble-meta");
    if (!metadata) return;
    const note = document.createElement("span");
    note.className = "sentiment-note";
    const name = model === "roberta" ? "RoBERTa" : "VADER";
    note.title = name + " estimates text polarity, not feelings or mental health.";
    note.textContent = " · Text sentiment (" + name + "): "
      + (score > 0 ? "+" : "") + score.toFixed(2);
    metadata.appendChild(note);
  }

  function appendMessage(role, message, options = {}) {
    const line = document.createElement("div");
    line.className = "message " + role;
    const avatar = document.createElement("span");
    avatar.className = "bubble-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = role === "user" ? "You" : "✳";
    const stack = document.createElement("div");
    stack.className = "bubble-stack";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = message;
    stack.appendChild(bubble);
    const meta = document.createElement("div");
    meta.className = "bubble-meta";
    const sourceNames = {
      gemini: "Gemini",
      dialogpt_fallback: "Local DialoGPT · Unreliable for medical guidance",
      safety_message: "Safety information",
      unavailable: "Service unavailable"
    };
    meta.textContent = role === "user" ? "You" : (sourceNames[options.model_used] || "MindSpace");
    stack.appendChild(meta);
    line.append(avatar, stack);
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
    return line;
  }

  function typingIndicator() {
    const line = document.createElement("div");
    line.className = "message bot";
    line.setAttribute("aria-label", "Preparing reply");
    const avatar = document.createElement("span");
    avatar.className = "bubble-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "✳";
    const stack = document.createElement("div");
    stack.className = "bubble-stack";
    const bubble = document.createElement("div");
    bubble.className = "bubble typing-bubble";
    bubble.setAttribute("aria-hidden", "true");
    for (let i = 0; i < 3; i++) bubble.appendChild(document.createElement("span"));
    stack.appendChild(bubble);
    line.append(avatar, stack);
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
    return line;
  }

  form.addEventListener("submit", async event => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message || send.disabled) return;
    input.value = "";
    fitInput();
    const userLine = appendMessage("user", message);
    send.disabled = true;
    input.disabled = true;
    status.textContent = "Thinking of a response…";
    const indicator = typingIndicator();
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 90000);
      let response;
      try {
        response = await fetch("/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({user_id: userId, message}),
          signal: controller.signal
        });
      } finally {
        clearTimeout(timeout);
      }
      const data = await response.json();
      if (!response.ok) {
        const detail = typeof data.detail === "string" ? data.detail : "The server could not process your message.";
        throw Error(detail);
      }
      addUserSentiment(userLine, data.sentiment_score, data.sentiment_model);
      indicator.remove();
      appendMessage("bot", String(data.reply || "No reply available."), {model_used: data.model_used});
      status.textContent = "Enter to send · Shift + Enter for a new line";
    } catch (error) {
      appendMessage("bot", error.name === "AbortError"
        ? "The response took too long. Please try again."
        : "Unable to send: " + error.message);
      status.textContent = "Could not complete the request";
    } finally {
      indicator.remove();
      send.disabled = false;
      input.disabled = false;
      input.focus();
    }
  });
  fitInput();
})();
