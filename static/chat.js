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

  function resourceLinks(stack, references, suggestedReading) {
    const grounded = Array.isArray(references) && references.length > 0;
    const links = grounded ? references : suggestedReading;
    if (!Array.isArray(links) || links.length === 0) return;
    const panel = document.createElement("div");
    panel.className = "source-box";
    const heading = document.createElement("p");
    heading.className = "source-heading";
    heading.textContent = grounded
      ? "Relevant resources · Not verification of every claim"
      : "Optional reading · Not used by the fallback model";
    panel.appendChild(heading);
    links.forEach(item => {
      if (!item || typeof item.url !== "string") return;
      try {
        const url = new URL(item.url);
        if (url.protocol !== "https:") return;
        const link = document.createElement("a");
        link.className = "source-link";
        link.href = url.href;
        link.textContent = String(item.title || url.hostname);
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        panel.appendChild(link);
      } catch (_) { /* Invalid resource URL is ignored */ }
    });
    if (panel.querySelector("a")) stack.appendChild(panel);
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
    if (role === "bot" && typeof options.sentiment_score === "number") {
      const score = document.createElement("span");
      score.textContent = " · Message sentiment: " + options.sentiment_score.toFixed(2);
      meta.appendChild(score);
    }
    stack.appendChild(meta);
    if (role === "bot") resourceLinks(stack, options.references, options.suggested_reading);
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
    appendMessage("user", message);
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
      appendMessage("bot", String(data.reply || "No reply available."), data);
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
