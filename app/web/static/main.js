// Server-side state owns the truth: this script is just a thin client that
// (a) submits messages, (b) subscribes to a job's SSE stream, (c) finalizes
// the visible message when the stream ends. Tab-away does not lose work
// — the server keeps generating; coming back resubscribes from the buffer.
(() => {
  const chatId = window.LOCALLLM.chatId;
  const initialJobId = window.LOCALLLM.activeJobId;

  const $messages = document.getElementById("messages");
  const $streaming = document.getElementById("streaming");
  const $streamingContent = $streaming.querySelector(".content");
  const $streamingMeta = $streaming.querySelector(".meta");
  const $form = document.getElementById("send-form");
  const $input = document.getElementById("composer-input");
  const $stopBtn = document.getElementById("stop-btn");

  const $modelSelection = document.getElementById("model-selection");
  const $useRag = document.getElementById("use-rag");
  const $systemPrompt = document.getElementById("system-prompt");
  const $temperature = document.getElementById("temperature");
  const $maxTokens = document.getElementById("max-tokens");
  const $numCtx = document.getElementById("num-ctx");

  let activeJobId = initialJobId;
  let activeSource = null;

  function scrollToBottom() {
    // Defer one frame so the just-appended node is laid out before we
    // measure scrollHeight (otherwise the scroll lags by one update).
    requestAnimationFrame(() => {
      $messages.scrollTop = $messages.scrollHeight;
    });
  }

  function isScrolledNearBottom() {
    return $messages.scrollHeight - $messages.scrollTop - $messages.clientHeight < 80;
  }

  function appendMessage(role, content) {
    const el = document.createElement("div");
    el.className = `message ${role}`;
    el.innerHTML = `<div class="role">${role}</div><div class="content"></div>`;
    el.querySelector(".content").textContent = content;
    $messages.appendChild(el);
    scrollToBottom();
    return el;
  }

  function showStreaming() {
    $streamingContent.textContent = "Waiting for first token…";
    $streamingContent.classList.add("placeholder");
    $streamingMeta.classList.add("hidden");
    $streamingMeta.textContent = "";
    $streaming.classList.remove("hidden");
    $messages.appendChild($streaming);
    scrollToBottom();
  }

  function hideStreaming() {
    $streaming.classList.add("hidden");
  }

  function finalizeStreaming(text, metadata, status) {
    if (text && text.trim()) {
      const finalText = (status === "stopped")
        ? `${text.trim()}\n\n_[stopped]_`
        : text;
      appendMessage("assistant", finalText);
    }
    hideStreaming();
  }

  function renderMeta(metadata) {
    if (!metadata) return;
    const parts = [];
    if (metadata.routing) {
      const r = metadata.routing;
      parts.push(`Auto-routed to ${r.selected_model} (score=${r.complexity_score})`);
    } else if (metadata.selected_model) {
      parts.push(`Model: ${metadata.selected_model}`);
    }
    if (metadata.rag_error) {
      parts.push(`⚠ RAG: ${metadata.rag_error}`);
    }
    if (metadata.retrievals && metadata.retrievals.length) {
      parts.push(`📚 ${metadata.retrievals.length} sources`);
    }
    if (parts.length) {
      $streamingMeta.textContent = parts.join("  ·  ");
      $streamingMeta.classList.remove("hidden");
    }
  }

  function subscribeToJob(jobId) {
    if (activeSource) {
      activeSource.close();
    }
    activeJobId = jobId;
    showStreaming();

    const src = new EventSource(`/api/jobs/${jobId}/stream`);
    activeSource = src;

    src.addEventListener("token", (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.chunk) {
          // Drop placeholder on first real token.
          if ($streamingContent.classList.contains("placeholder")) {
            $streamingContent.textContent = "";
            $streamingContent.classList.remove("placeholder");
          }
          const wasNearBottom = isScrolledNearBottom();
          $streamingContent.textContent += data.chunk;
          if (wasNearBottom) scrollToBottom();
        }
      } catch (err) {
        console.warn("token parse error", err, e.data);
      }
    });

    src.addEventListener("done", (e) => {
      try {
        const data = JSON.parse(e.data);
        renderMeta(data.metadata);
        finalizeStreaming(data.text || $streamingContent.textContent, data.metadata, data.status);
      } finally {
        src.close();
        activeSource = null;
        activeJobId = null;
      }
    });

    src.onerror = () => {
      // Browser will automatically retry; if the server has finished and
      // gone away, the next replay will catch us up.
      console.warn("EventSource error; will retry");
    };
  }

  // -------------------- Form submission --------------------

  $form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const content = $input.value.trim();
    if (!content) return;

    // Echo user message immediately
    appendMessage("user", content);
    $input.value = "";

    const body = {
      content,
      model_selection: $modelSelection.value,
      use_rag: $useRag.checked,
      system_prompt: $systemPrompt.value,
      temperature: parseFloat($temperature.value),
      max_tokens: parseInt($maxTokens.value, 10),
      num_ctx: parseInt($numCtx.value, 10),
    };

    try {
      const res = await fetch(`/api/chats/${chatId}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const text = await res.text();
        appendMessage("assistant", `Error: ${res.status} ${text}`);
        return;
      }
      const data = await res.json();
      subscribeToJob(data.job_id);
    } catch (err) {
      appendMessage("assistant", `Error: ${err}`);
    }
  });

  // Enter to send, Shift+Enter newline.
  $input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $form.requestSubmit();
    }
  });

  // -------------------- Stop --------------------

  $stopBtn.addEventListener("click", async () => {
    if (!activeJobId) return;
    try {
      await fetch(`/api/jobs/${activeJobId}/stop`, { method: "POST" });
    } catch (err) {
      console.warn("stop failed", err);
    }
  });

  // -------------------- Delete chat --------------------

  document.querySelectorAll(".btn.del").forEach(btn => {
    btn.addEventListener("click", async () => {
      const cid = btn.dataset.chatId;
      if (!confirm("Delete this chat?")) return;
      try {
        await fetch(`/chats/${cid}`, { method: "DELETE" });
        if (cid === chatId) {
          window.location.href = "/";
        } else {
          btn.closest(".chat-row").remove();
        }
      } catch (err) {
        console.warn("delete failed", err);
      }
    });
  });

  // -------------------- Resume in-flight job on load --------------------

  if (initialJobId) {
    subscribeToJob(initialJobId);
  }

  scrollToBottom();
})();
