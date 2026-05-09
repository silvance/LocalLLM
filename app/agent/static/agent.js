// Online-agent page client. Kicks off /api/agent, subscribes to the
// returned job's SSE stream, and renders each event as a labeled card.
(() => {
  const $promptInput = document.getElementById("agent-prompt-input");
  const $runBtn = document.getElementById("run-agent");
  const $stream = document.getElementById("agent-stream");
  const $status = document.getElementById("agent-status");
  const $model = document.getElementById("agent-model");
  const loadedAgent = window.LOCALLLM && window.LOCALLLM.loadedAgent;

  let activeJobId = null;
  let activeSource = null;
  let currentAgentId = (loadedAgent && loadedAgent.id) || null;

  function escapeHtml(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function scrollToBottom() {
    requestAnimationFrame(() => {
      $stream.scrollTop = $stream.scrollHeight;
    });
  }

  function isScrolledNearBottom() {
    return $stream.scrollHeight - $stream.scrollTop - $stream.clientHeight < 120;
  }

  function appendEvent(kind, headerLabel, bodyHtml, extra = {}) {
    const wasNearBottom = isScrolledNearBottom();
    const ev = document.createElement("details");
    ev.className = `agent-event kind-${kind}`;
    // Default open; user can click the header (the <summary>) to fold.
    ev.open = true;
    ev.innerHTML =
      `<summary class="event-header"><span>${escapeHtml(headerLabel)}</span>${extra.right || ""}</summary>` +
      `<div class="event-body">${bodyHtml}</div>`;
    $stream.appendChild(ev);
    if (wasNearBottom) scrollToBottom();
    return ev;
  }

  function renderAgentEvent(kind, payload) {
    const d = payload || {};
    if (kind === "user_prompt") {
      appendEvent("user", "User", escapeHtml(d.prompt || "").replace(/\n/g, "<br>"));
      return;
    }
    if (kind === "step_start") {
      appendEvent("step", `Step ${(d.iteration || 0) + 1}`, '<span class="muted-note">model is thinking…</span>');
      return;
    }
    if (kind === "model_text") {
      appendEvent("step", "Reasoning", escapeHtml(d.content || "").replace(/\n/g, "<br>"));
      return;
    }
    if (kind === "tool_call") {
      const argsJson = JSON.stringify(d.args || {}, null, 2);
      appendEvent(
        "tool-call",
        `→ tool: ${d.name || ""}`,
        `<details open><summary>args</summary><pre>${escapeHtml(argsJson)}</pre></details>`,
      );
      return;
    }
    if (kind === "tool_result") {
      let body;
      if (d.name === "web_search" && d.result && Array.isArray(d.result.results)) {
        body = renderSearchResults(d.result.results);
      } else if (d.name === "http_fetch" && d.result && typeof d.result.text === "string") {
        body = renderFetchResult(d.result);
      } else {
        body = `<pre>${escapeHtml(JSON.stringify(d.result || {}, null, 2))}</pre>`;
      }
      appendEvent("tool-result", `← result: ${d.name || ""}`, body);
      return;
    }
    if (kind === "tool_error") {
      appendEvent("tool-error", `× tool failed: ${d.name || ""}`, renderToolError(d.error || ""));
      return;
    }
    if (kind === "error") {
      appendEvent("error", "ERROR", escapeHtml(d.error || ""));
      return;
    }
    if (kind === "final") {
      appendEvent(
        "final",
        `Final answer (${escapeHtml(d.model || "")}, ${escapeHtml(d.status || "done")})`,
        escapeHtml(d.answer || "").replace(/\n/g, "<br>"),
      );
      return;
    }
    appendEvent("step", kind || "Event", `<pre>${escapeHtml(JSON.stringify(d, null, 2))}</pre>`);
  }

  function renderSearchResults(results) {
    if (!results || !results.length) return '<span class="muted-note">no results</span>';
    return results.map(r =>
      `<div class="search-result">` +
        `<div class="search-result-title">${escapeHtml(r.title || "(untitled)")}</div>` +
        `<div class="search-result-url"><a href="${escapeHtml(r.url || "#")}" target="_blank" rel="noreferrer">${escapeHtml(r.url || "")}</a></div>` +
        `<div class="search-result-snippet">${escapeHtml(r.snippet || "")}</div>` +
      `</div>`
    ).join("");
  }

  // Pull a `pip install` (or similar) command out of an error message so we
  // can render it as a copy-friendly hint. The web_search / http_fetch
  // tools format their import-error messages as: "...; run `pip install
  // -r requirements-agent.txt`" — match that pattern, but be lenient.
  function extractInstallHint(text) {
    const m = String(text || "").match(/run\s+`([^`]+)`/i);
    return m ? m[1] : "";
  }

  function renderToolError(error) {
    const hint = extractInstallHint(error);
    let body = escapeHtml(error || "(no detail)");
    if (hint) {
      body +=
        `<div class="install-hint">` +
          `<span>Install hint:</span>` +
          `<code>${escapeHtml(hint)}</code>` +
        `</div>`;
    }
    return body;
  }

  function renderFetchResult(r) {
    const meta = `<div class="fetch-meta">HTTP ${escapeHtml(r.status)} · ${escapeHtml(r.chars)} chars${r.truncated ? " (truncated)" : ""}` +
                 (r.title ? ` · ${escapeHtml(r.title)}` : "") + `</div>`;
    const preview = (r.text || "").slice(0, 600);
    const tail = (r.text || "").length > 600 ? "…" : "";
    return meta +
      `<details><summary>Show full extracted text (${escapeHtml(r.chars)} chars)</summary>` +
        `<pre>${escapeHtml(r.text || "")}</pre>` +
      `</details>` +
      `<div style="margin-top:0.5rem;">${escapeHtml(preview)}${tail}</div>`;
  }

  function setStatus(text, color) {
    $status.textContent = text;
    $status.style.color = color || "";
  }

  function clearStream() {
    $stream.innerHTML = "";
  }

  function ensureHistoryRow(agentId, title, status) {
    const list = document.querySelector(".sessions .chat-list");
    if (!list || !agentId) return;
    const existing = list.querySelector(`[data-agent-id="${CSS.escape(agentId)}"]`);
    if (existing) return;
    const row = document.createElement("li");
    row.className = "chat-row active";
    row.innerHTML =
      `<a href="/agent/${escapeHtml(agentId)}" class="chat-link" title="${escapeHtml(title || "New agent run")}">` +
        `${escapeHtml(title || "New agent run")}` +
        `<span class="count">${escapeHtml(status || "running")}</span>` +
      `</a>` +
      `<button type="button" class="btn ghost del-agent" data-agent-id="${escapeHtml(agentId)}" title="Delete this agent run">✕</button>`;
    list.prepend(row);
  }

  function renderLoadedSession(session) {
    if (!session) return;
    clearStream();
    $promptInput.value = "";
    if (session.model && Array.from($model.options).some((o) => o.value === session.model)) {
      $model.value = session.model;
    }
    const events = session.events || [];
    const hasUserPrompt = events.some((ev) => ev.kind === "user_prompt");
    const hasFinal = events.some((ev) => ev.kind === "final");
    if (!hasUserPrompt && (session.prompt || "").trim()) {
      renderAgentEvent("user_prompt", { prompt: session.prompt });
    }
    for (const ev of events) {
      renderAgentEvent(ev.kind, ev.payload || {});
    }
    if (!hasFinal && (session.answer || "").trim()) {
      appendEvent(
        "final",
        `Final answer (${escapeHtml(session.model || "")}, ${escapeHtml(session.status || "done")})`,
        escapeHtml(session.answer || "").replace(/\n/g, "<br>"),
      );
    } else if (session.error) {
      appendEvent("error", "ERROR", escapeHtml(session.error));
    } else if (session.events && session.events.length) {
      appendEvent("final", `Saved run (${escapeHtml(session.status || "pending")})`, '<span class="muted-note">no final answer was saved</span>');
    }
    setStatus(session.status ? `saved (${session.status})` : "saved", "");
  }

  function startAgent() {
    const prompt = $promptInput.value.trim();
    if (!prompt) return;

    if (!loadedAgent) clearStream();
    setStatus("starting…", "");
    $runBtn.disabled = true;

    const endpoint = currentAgentId
      ? `/api/agent/${encodeURIComponent(currentAgentId)}/messages`
      : "/api/agent";
    fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, model: $model.value }),
    })
      .then(r => {
        if (!r.ok) return r.text().then(t => { throw new Error(`${r.status}: ${t}`); });
        return r.json();
      })
      .then(data => {
        if (!data.job_id) throw new Error("no job id returned");
        $promptInput.value = "";
        if (data.agent_id) {
          currentAgentId = data.agent_id;
          $runBtn.textContent = "▶ Send follow-up";
        }
        if (endpoint === "/api/agent" && data.agent_id) {
          history.replaceState(null, "", `/agent/${data.agent_id}`);
          ensureHistoryRow(data.agent_id, data.title, "running");
        }
        subscribe(data.job_id);
      })
      .catch(err => {
        setStatus(`error: ${err}`, "var(--warn)");
        $runBtn.disabled = false;
        appendEvent("error", "ERROR", escapeHtml(String(err)));
      });
  }

  function subscribe(jobId) {
    activeJobId = jobId;
    if (activeSource) activeSource.close();
    const src = new EventSource(`/api/jobs/${jobId}/stream`);
    activeSource = src;
    setStatus("running…", "");

    src.addEventListener("step_start", (e) => renderAgentEvent("step_start", JSON.parse(e.data)));
    src.addEventListener("user_prompt", (e) => renderAgentEvent("user_prompt", JSON.parse(e.data)));
    src.addEventListener("model_text", (e) => renderAgentEvent("model_text", JSON.parse(e.data)));
    src.addEventListener("tool_call", (e) => renderAgentEvent("tool_call", JSON.parse(e.data)));
    src.addEventListener("tool_result", (e) => renderAgentEvent("tool_result", JSON.parse(e.data)));
    src.addEventListener("tool_error", (e) => renderAgentEvent("tool_error", JSON.parse(e.data)));

    src.addEventListener("error", (e) => {
      try {
        renderAgentEvent("error", JSON.parse(e.data));
      } catch (_) { /* native EventSource error event has no data */ }
    });

    src.addEventListener("done", (e) => {
      try {
        const d = JSON.parse(e.data);
        const meta = d.metadata || {};
        const status = d.status || "done";
        const answer = (meta.answer || "").trim();
        if (answer) {
          appendEvent(
            "final",
            `Final answer (${escapeHtml(meta.model || "")}, ${status})`,
            escapeHtml(answer).replace(/\n/g, "<br>"),
          );
        } else {
          appendEvent("final", `Done (${status})`, '<span class="muted-note">no final answer was produced</span>');
        }
        setStatus(`done (${status})`, "");
      } finally {
        src.close();
        activeSource = null;
        activeJobId = null;
        $runBtn.disabled = false;
      }
    });

    src.onerror = () => console.warn("EventSource error; will retry");
  }

  $runBtn.addEventListener("click", startAgent);
  $promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      startAgent();
    }
  });

  const $sessions = document.querySelector(".sessions");
  if ($sessions) {
    $sessions.addEventListener("click", async (e) => {
      const btn = e.target.closest(".del-agent");
      if (!btn) return;
      const id = btn.dataset.agentId;
      if (!id || !confirm("Delete this saved agent run?")) return;
      try {
        await fetch(`/agent/${id}`, { method: "DELETE" });
        const loadedId = (window.LOCALLLM && window.LOCALLLM.loadedAgent && window.LOCALLLM.loadedAgent.id) || "";
        if (loadedId === id) {
          window.location.href = "/agent";
        } else {
          btn.closest(".chat-row").remove();
        }
      } catch {}
    });
  }

  // Resume on reload: server passed an in-progress job, reattach the
  // SSE stream + repopulate the prompt textarea so the page mirrors
  // what was running before navigation.
  const activeJob = window.LOCALLLM && window.LOCALLLM.activeJob;
  if (activeJob && activeJob.job_id) {
    if (activeJob.agent_id) currentAgentId = activeJob.agent_id;
    $promptInput.value = activeJob.prompt || "";
    if (activeJob.model) {
      const modelSelect = document.getElementById("agent-model");
      if (modelSelect && Array.from(modelSelect.options).some((o) => o.value === activeJob.model)) {
        modelSelect.value = activeJob.model;
      }
    }
    setStatus("resuming…", "");
    $runBtn.disabled = true;
    subscribe(activeJob.job_id);
  } else if (loadedAgent) {
    renderLoadedSession(loadedAgent);
  }
})();
