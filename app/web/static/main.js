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
  const $copyChatBtn = document.getElementById("copy-chat-btn");

  const $modelSelection = document.getElementById("model-selection");
  const $useRag = document.getElementById("use-rag");
  const $systemPrompt = document.getElementById("system-prompt");
  const $temperature = document.getElementById("temperature");
  const $maxTokens = document.getElementById("max-tokens");
  const $numCtx = document.getElementById("num-ctx");

  let activeJobId = initialJobId;
  let activeSource = null;

  // ============================================================
  // Helpers
  // ============================================================

  function escapeHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // Tiny markdown-ish renderer: only handles fenced code blocks (```lang ... ```).
  // Everything else stays as plain text. Returns HTML.
  function renderWithCodeBlocks(text) {
    const out = [];
    const fence = /```([A-Za-z0-9_+-]*)\r?\n?([\s\S]*?)```/g;
    let last = 0;
    let m;
    while ((m = fence.exec(text)) !== null) {
      if (m.index > last) {
        out.push(`<span class="prose">${escapeHtml(text.slice(last, m.index))}</span>`);
      }
      const lang = m[1] || "";
      const code = m[2] || "";
      out.push(
        `<div class="code-block">` +
          `<div class="code-header">` +
            `<span class="code-lang">${escapeHtml(lang || "code")}</span>` +
            `<button type="button" class="btn ghost copy-code">📋 Copy</button>` +
          `</div>` +
          `<pre><code class="language-${escapeHtml(lang)}">${escapeHtml(code)}</code></pre>` +
        `</div>`,
      );
      last = fence.lastIndex;
    }
    if (last < text.length) {
      out.push(`<span class="prose">${escapeHtml(text.slice(last))}</span>`);
    }
    return out.join("");
  }

  async function copyToClipboard(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (err) {
      // Older browsers / non-secure contexts fall back to a hidden textarea.
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      let ok = false;
      try { ok = document.execCommand("copy"); } catch (_) { ok = false; }
      document.body.removeChild(ta);
      return ok;
    }
  }

  function flashCopied(btn, label = "Copied") {
    const original = btn.textContent;
    btn.textContent = `✓ ${label}`;
    btn.classList.add("copied");
    setTimeout(() => {
      btn.textContent = original;
      btn.classList.remove("copied");
    }, 1200);
  }

  // ============================================================
  // Message rendering
  // ============================================================

  function buildMessageElement(role, content) {
    const el = document.createElement("div");
    el.className = `message ${role}`;
    el.dataset.role = role;
    const actionBtn = role === "user"
      ? `<button type="button" class="btn ghost edit-msg" title="Edit and re-run">✎</button>`
      : role === "assistant"
        ? `<button type="button" class="btn ghost regen-msg" title="Regenerate this response">↻</button>`
        : "";
    el.innerHTML =
      `<div class="message-header">` +
        `<span class="role">${role}</span>` +
        `<button type="button" class="btn ghost copy-msg" title="Copy message text">📋</button>` +
        actionBtn +
      `</div>` +
      `<div class="content"></div>`;
    const contentEl = el.querySelector(".content");
    contentEl.dataset.raw = content;
    if (role === "assistant") {
      contentEl.innerHTML = renderWithCodeBlocks(content);
    } else {
      contentEl.textContent = content;
    }
    return el;
  }

  // Body shared by send / regenerate / edit. Pulls from the sidebar
  // controls so a request always reflects the current settings.
  function currentRequestBody(extra) {
    return Object.assign({
      model_selection: $modelSelection.value,
      use_rag: $useRag.checked,
      system_prompt: $systemPrompt.value,
      temperature: parseFloat($temperature.value),
      max_tokens: parseInt($maxTokens.value, 10),
      num_ctx: parseInt($numCtx.value, 10),
    }, extra || {});
  }

  function messageIndex(messageEl) {
    return Array.from($messages.querySelectorAll(".message")).indexOf(messageEl);
  }

  function removeMessagesFrom(index) {
    Array.from($messages.querySelectorAll(".message"))
      .slice(index)
      .forEach((el) => el.remove());
  }

  async function regenerateAt(index) {
    if (activeJobId) {
      // Don't pile a regen on top of an in-flight stream.
      return;
    }
    try {
      const res = await fetch(
        `/api/chats/${chatId}/messages/${index}/regenerate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(currentRequestBody()),
        },
      );
      if (!res.ok) {
        const text = await res.text();
        appendMessage("assistant", `Error: ${res.status} ${text}`);
        return;
      }
      const data = await res.json();
      // Drop the assistant message we're about to replace, then attach
      // the new stream into the existing live-preview slot.
      removeMessagesFrom(index);
      subscribeToJob(data.job_id);
    } catch (err) {
      appendMessage("assistant", `Error: ${err}`);
    }
  }

  function startEditAt(index, messageEl) {
    if (messageEl.querySelector(".edit-form")) return;
    const contentEl = messageEl.querySelector(".content");
    const original = contentEl.dataset.raw || contentEl.textContent;
    const form = document.createElement("form");
    form.className = "edit-form";
    form.innerHTML =
      `<textarea class="edit-input" rows="3"></textarea>` +
      `<div class="edit-actions">` +
        `<button type="button" class="btn ghost edit-cancel">Cancel</button>` +
        `<button type="submit" class="btn primary">Save and re-run</button>` +
      `</div>`;
    const textarea = form.querySelector(".edit-input");
    textarea.value = original;
    contentEl.classList.add("hidden");
    messageEl.appendChild(form);
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);

    const cleanup = () => {
      form.remove();
      contentEl.classList.remove("hidden");
    };

    form.querySelector(".edit-cancel").addEventListener("click", cleanup);

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const newText = textarea.value.trim();
      if (!newText) return;
      if (activeJobId) return;
      try {
        const res = await fetch(
          `/api/chats/${chatId}/messages/${index}/edit`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(currentRequestBody({ content: newText })),
          },
        );
        if (!res.ok) {
          const text = await res.text();
          appendMessage("assistant", `Error: ${res.status} ${text}`);
          return;
        }
        const data = await res.json();
        // Replace the user message in place + drop everything after,
        // then start streaming the new response.
        contentEl.dataset.raw = newText;
        contentEl.textContent = newText;
        cleanup();
        removeMessagesFrom(index + 1);
        subscribeToJob(data.job_id);
      } catch (err) {
        appendMessage("assistant", `Error: ${err}`);
      }
    });
  }

  function appendMessage(role, content) {
    // First message in a fresh chat: drop the centered empty-state
    // welcome so the conversation flows from the top instead of
    // showing the welcome text shoved up against the new bubble.
    const welcome = $messages.querySelector(".empty-welcome");
    if (welcome) welcome.remove();
    const el = buildMessageElement(role, content);
    $messages.appendChild(el);
    scrollToBottom();
    return el;
  }

  // Re-process server-rendered messages on initial load: hydrate raw text into
  // properly rendered HTML (code blocks etc.) and wire up copy buttons.
  function hydrateInitialMessages() {
    document.querySelectorAll(".messages .message").forEach((el) => {
      const role = el.dataset.role;
      const content = el.querySelector(".content");
      const raw = content.dataset.raw || content.textContent;
      content.dataset.raw = raw;
      if (role === "assistant") {
        content.innerHTML = renderWithCodeBlocks(raw);
      } else {
        content.textContent = raw;
      }
    });
  }

  // ============================================================
  // Streaming
  // ============================================================

  // The chat column owns the scroll (so the scrollbar sits at the
  // column edge, not at the centered .messages edge). Read/write
  // scroll on the closest scrollable ancestor of $messages so we
  // don't need to know whether it's <main class="chat"> or some
  // future wrapper.
  const $scroller = $messages.parentElement || $messages;

  function scrollToBottom() {
    requestAnimationFrame(() => {
      $scroller.scrollTop = $scroller.scrollHeight;
    });
  }

  function isScrolledNearBottom() {
    return $scroller.scrollHeight - $scroller.scrollTop - $scroller.clientHeight < 80;
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
      const el = appendMessage("assistant", finalText);
      attachCitations(el, metadata);
    }
    hideStreaming();
  }

  // Citation panel attached below an assistant message. Not persisted —
  // a page refresh drops them (the chat storage only holds role/content),
  // which is fine since the user just saw them stream in.
  function attachCitations(messageEl, metadata) {
    const retrievals = metadata && metadata.retrievals;
    if (!retrievals || !retrievals.length) return;
    const panel = document.createElement("details");
    panel.className = "citations";
    const summary = document.createElement("summary");
    summary.textContent = `📚 ${retrievals.length} source${retrievals.length === 1 ? "" : "s"}`;
    panel.appendChild(summary);
    const list = document.createElement("ol");
    retrievals.forEach((r) => {
      const li = document.createElement("li");
      const head = document.createElement("div");
      head.className = "citation-head";
      const path = document.createElement("code");
      path.textContent = `${r.source_id}/${r.file_path}`;
      head.appendChild(path);
      if (typeof r.score === "number") {
        const score = document.createElement("span");
        score.className = "citation-score";
        score.textContent = `score=${r.score.toFixed(2)}`;
        head.appendChild(score);
      }
      li.appendChild(head);
      if (r.snippet) {
        const pre = document.createElement("pre");
        pre.className = "citation-snippet";
        pre.textContent = r.snippet;
        li.appendChild(pre);
      }
      list.appendChild(li);
    });
    panel.appendChild(list);
    messageEl.appendChild(panel);
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
          if ($streamingContent.classList.contains("placeholder")) {
            $streamingContent.textContent = "";
            $streamingContent.classList.remove("placeholder");
          }
          const wasNearBottom = isScrolledNearBottom();
          // Streaming bubble stays plain text; we render code blocks on
          // finalize so we don't try to parse half-streamed fences.
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
      console.warn("EventSource error; will retry");
    };
  }

  // ============================================================
  // Slash commands — Discord-style local UI controls
  // ============================================================

  // Each command runs in the browser only. None of them send the
  // entered text to the model; they tweak UI state, trigger an
  // existing action (regen/edit/copy), or navigate. Unknown `/foo`
  // input falls through to the normal send path so the model still
  // sees it (e.g. someone literally asking about a "/clear" syntax).
  function lastMessageIndex(role) {
    const all = Array.from($messages.querySelectorAll(".message"));
    for (let i = all.length - 1; i >= 0; i--) {
      if (!role || all[i].dataset.role === role) return i;
    }
    return -1;
  }

  function showSystemMessage(html) {
    // Local, unpersisted "system" bubble — not part of the chat history.
    const el = document.createElement("div");
    el.className = "message system local";
    el.dataset.role = "system";
    el.innerHTML =
      `<div class="message-header"><span class="role">system</span></div>` +
      `<div class="content"></div>`;
    el.querySelector(".content").innerHTML = html;
    $messages.appendChild(el);
    scrollToBottom();
  }

  function htmlEscape(s) {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // Order matters in the help list — keep it alphabetical for readability.
  const SLASH_COMMANDS = {
    "help": {
      args: "",
      desc: "List slash commands",
      run: () => {
        const rows = Object.entries(SLASH_COMMANDS)
          .map(([name, c]) => {
            const usage = `/${name}${c.args ? " " + c.args : ""}`;
            return `<tr><td><code>${htmlEscape(usage)}</code></td><td>${htmlEscape(c.desc)}</td></tr>`;
          }).join("");
        showSystemMessage(`<table class="slash-help"><tbody>${rows}</tbody></table>`);
      },
    },
    "clear": {
      args: "",
      desc: "Start a fresh chat (current chat is preserved on disk)",
      run: async () => {
        // POST /chats creates and redirects; just submit the form that's
        // already on the page so we follow the same redirect path.
        const form = document.querySelector('form[action="/chats"]');
        if (form) form.submit();
      },
    },
    "new": {
      args: "",
      desc: "Alias for /clear",
      run: () => SLASH_COMMANDS.clear.run(),
    },
    "model": {
      args: "[auto|granite|gemma|qwen]",
      desc: "Show or set the active model",
      run: (arg) => {
        if (!arg) {
          showSystemMessage(`Current model: <code>${htmlEscape($modelSelection.value)}</code>`);
          return;
        }
        const allowed = Array.from($modelSelection.options).map((o) => o.value);
        if (!allowed.includes(arg)) {
          showSystemMessage(`Unknown model <code>${htmlEscape(arg)}</code>. Options: ${allowed.map((m) => `<code>${m}</code>`).join(", ")}`);
          return;
        }
        $modelSelection.value = arg;
        showSystemMessage(`Model set to <code>${htmlEscape(arg)}</code>`);
      },
    },
    "rag": {
      args: "[on|off|toggle]",
      desc: "Show or change the RAG retrieval toggle",
      run: (arg) => {
        const cur = $useRag.checked;
        const next = !arg
          ? null
          : arg === "on"     ? true
          : arg === "off"    ? false
          : arg === "toggle" ? !cur
          : null;
        if (next === null && arg) {
          showSystemMessage(`Usage: <code>/rag on|off|toggle</code>. Current: <code>${cur ? "on" : "off"}</code>`);
          return;
        }
        if (next === null) {
          showSystemMessage(`RAG is <code>${cur ? "on" : "off"}</code>`);
          return;
        }
        $useRag.checked = next;
        showSystemMessage(`RAG set to <code>${next ? "on" : "off"}</code>`);
      },
    },
    "system": {
      args: "[prompt text]",
      desc: "Show or set the system prompt",
      run: (arg) => {
        if (!arg) {
          const cur = ($systemPrompt.value || "").trim() || "(empty)";
          showSystemMessage(`<details><summary>System prompt (${cur.length} chars)</summary><pre>${htmlEscape(cur)}</pre></details>`);
          return;
        }
        $systemPrompt.value = arg;
        showSystemMessage(`System prompt set (${arg.length} chars)`);
      },
    },
    "regen": {
      args: "",
      desc: "Regenerate the last assistant response",
      run: () => {
        const idx = lastMessageIndex("assistant");
        if (idx < 0) {
          showSystemMessage("No assistant message to regenerate.");
          return;
        }
        regenerateAt(idx);
      },
    },
    "regenerate": {
      args: "",
      desc: "Alias for /regen",
      run: () => SLASH_COMMANDS.regen.run(),
    },
    "edit": {
      args: "<new content>",
      desc: "Edit the last user message and re-run from there",
      run: (arg) => {
        if (!arg) {
          showSystemMessage("Usage: <code>/edit your new message text</code>");
          return;
        }
        const idx = lastMessageIndex("user");
        if (idx < 0) {
          showSystemMessage("No user message to edit.");
          return;
        }
        // Reuse the same path as the inline ✎ button: open editor + submit.
        const all = Array.from($messages.querySelectorAll(".message"));
        const messageEl = all[idx];
        startEditAt(idx, messageEl);
        const textarea = messageEl.querySelector(".edit-input");
        if (textarea) textarea.value = arg;
        const form = messageEl.querySelector(".edit-form");
        if (form) form.requestSubmit();
      },
    },
    "copy": {
      args: "",
      desc: "Copy the last assistant response to the clipboard",
      run: async () => {
        const idx = lastMessageIndex("assistant");
        if (idx < 0) {
          showSystemMessage("No assistant message to copy.");
          return;
        }
        const all = Array.from($messages.querySelectorAll(".message"));
        const content = all[idx].querySelector(".content");
        const raw = content.dataset.raw || content.textContent;
        const ok = await copyToClipboard(raw);
        showSystemMessage(ok ? "Copied to clipboard." : "Copy failed.");
      },
    },
  };

  // Returns true if `text` was a known slash command and was handled.
  // Returns false for unknown `/foo` so it falls through to send.
  async function tryHandleSlashCommand(text) {
    if (!text.startsWith("/")) return false;
    const space = text.indexOf(" ");
    const name = (space === -1 ? text.slice(1) : text.slice(1, space)).toLowerCase();
    const arg = (space === -1 ? "" : text.slice(space + 1)).trim();
    const cmd = SLASH_COMMANDS[name];
    if (!cmd) return false;
    try {
      await cmd.run(arg);
    } catch (err) {
      showSystemMessage(`Command <code>/${htmlEscape(name)}</code> failed: ${htmlEscape(String(err))}`);
    }
    return true;
  }

  // ============================================================
  // Form submission
  // ============================================================

  $form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const content = $input.value.trim();
    if (!content) return;

    if (await tryHandleSlashCommand(content)) {
      $input.value = "";
      return;
    }

    appendMessage("user", content);
    $input.value = "";

    try {
      const res = await fetch(`/api/chats/${chatId}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(currentRequestBody({ content })),
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

  $input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $form.requestSubmit();
    }
  });

  // ============================================================
  // Stop / delete chat
  // ============================================================

  $stopBtn.addEventListener("click", async () => {
    if (!activeJobId) return;
    try {
      await fetch(`/api/jobs/${activeJobId}/stop`, { method: "POST" });
    } catch (err) {
      console.warn("stop failed", err);
    }
  });

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

  // ============================================================
  // Copy actions: per-message, per-code-block, whole chat
  // ============================================================

  // Event delegation so dynamically-added messages get the same behavior.
  $messages.addEventListener("click", async (e) => {
    const target = e.target;
    if (target.classList.contains("copy-msg")) {
      const msg = target.closest(".message");
      const content = msg.querySelector(".content");
      const raw = content.dataset.raw || content.textContent;
      const ok = await copyToClipboard(raw);
      if (ok) flashCopied(target);
      return;
    }
    if (target.classList.contains("copy-code")) {
      const block = target.closest(".code-block");
      const codeEl = block.querySelector("pre code");
      const ok = await copyToClipboard(codeEl.textContent);
      if (ok) flashCopied(target);
      return;
    }
    if (target.classList.contains("regen-msg")) {
      const msg = target.closest(".message");
      const idx = messageIndex(msg);
      if (idx >= 0) regenerateAt(idx);
      return;
    }
    if (target.classList.contains("edit-msg")) {
      const msg = target.closest(".message");
      const idx = messageIndex(msg);
      if (idx >= 0) startEditAt(idx, msg);
      return;
    }
  });

  $copyChatBtn?.addEventListener("click", async () => {
    const parts = [];
    document.querySelectorAll(".messages .message").forEach((el) => {
      const role = el.dataset.role || "unknown";
      const raw = el.querySelector(".content").dataset.raw || el.querySelector(".content").textContent;
      const heading = role.charAt(0).toUpperCase() + role.slice(1);
      parts.push(`## ${heading}\n\n${raw.trim()}\n`);
    });
    const ok = await copyToClipboard(parts.join("\n"));
    if (ok) flashCopied($copyChatBtn, "Chat copied");
  });

  // ============================================================
  // Init
  // ============================================================

  hydrateInitialMessages();

  if (initialJobId) {
    subscribeToJob(initialJobId);
  }

  scrollToBottom();
})();
