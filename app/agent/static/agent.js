// Online-agent page client. Kicks off /api/agent, subscribes to the
// returned job's SSE stream, and renders each event as a labeled card.
(() => {
  const $promptInput = document.getElementById("agent-prompt-input");
  const $runBtn = document.getElementById("run-agent");
  const $stream = document.getElementById("agent-stream");
  const $status = document.getElementById("agent-status");
  const $model = document.getElementById("agent-model");

  let activeJobId = null;
  let activeSource = null;

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
    const ev = document.createElement("div");
    ev.className = `agent-event kind-${kind}`;
    ev.innerHTML =
      `<div class="event-header"><span>${escapeHtml(headerLabel)}</span>${extra.right || ""}</div>` +
      `<div class="event-body">${bodyHtml}</div>`;
    $stream.appendChild(ev);
    if (wasNearBottom) scrollToBottom();
    return ev;
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

  function startAgent() {
    const prompt = $promptInput.value.trim();
    if (!prompt) return;

    clearStream();
    setStatus("starting…", "");
    $runBtn.disabled = true;

    fetch("/api/agent", {
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

    src.addEventListener("step_start", (e) => {
      const d = JSON.parse(e.data);
      appendEvent("step", `Step ${d.iteration + 1}`, '<span class="muted-note">model is thinking…</span>');
    });

    src.addEventListener("model_text", (e) => {
      const d = JSON.parse(e.data);
      appendEvent("step", "Reasoning", escapeHtml(d.content || "").replace(/\n/g, "<br>"));
    });

    src.addEventListener("tool_call", (e) => {
      const d = JSON.parse(e.data);
      const argsJson = JSON.stringify(d.args || {}, null, 2);
      appendEvent(
        "tool-call",
        `→ tool: ${d.name}`,
        `<details open><summary>args</summary><pre>${escapeHtml(argsJson)}</pre></details>`,
      );
    });

    src.addEventListener("tool_result", (e) => {
      const d = JSON.parse(e.data);
      let body;
      if (d.name === "web_search" && d.result && Array.isArray(d.result.results)) {
        body = renderSearchResults(d.result.results);
      } else if (d.name === "http_fetch" && d.result && typeof d.result.text === "string") {
        body = renderFetchResult(d.result);
      } else {
        body = `<pre>${escapeHtml(JSON.stringify(d.result || {}, null, 2))}</pre>`;
      }
      appendEvent("tool-result", `← result: ${d.name}`, body);
    });

    src.addEventListener("tool_error", (e) => {
      const d = JSON.parse(e.data);
      appendEvent("tool-error", `× tool failed: ${d.name}`, escapeHtml(d.error || ""));
    });

    src.addEventListener("error", (e) => {
      try {
        const d = JSON.parse(e.data);
        appendEvent("error", "ERROR", escapeHtml(d.error || ""));
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
})();
