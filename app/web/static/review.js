// Adversarial review: writer ↔ reviewer ↔ writer loop. The server runs
// the whole loop as a single Job; this client just opens an SSE stream
// and renders sections delimited by `section_start` events.
(() => {
  const $promptInput = document.getElementById("review-prompt-input");
  const $runBtn = document.getElementById("run-review");
  const $stopBtn = document.getElementById("stop-review");
  const $stream = document.getElementById("review-stream");

  const $writerModel = document.getElementById("writer-model");
  const $reviewerModel = document.getElementById("reviewer-model");
  const $rounds = document.getElementById("rounds");
  const $systemPrompt = document.getElementById("system-prompt");
  const $temperature = document.getElementById("temperature");
  const $maxTokens = document.getElementById("max-tokens");
  const $numCtx = document.getElementById("num-ctx");

  let activeJobId = null;
  let activeSource = null;
  let currentSection = null;  // DOM element for the in-progress section

  // ---- helpers (small subset of main.js so the page stands alone) ----

  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;")
            .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function renderWithCodeBlocks(text) {
    const out = [];
    const fence = /```([A-Za-z0-9_+-]*)\r?\n?([\s\S]*?)```/g;
    let last = 0, m;
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
        `</div>`
      );
      last = fence.lastIndex;
    }
    if (last < text.length) {
      out.push(`<span class="prose">${escapeHtml(text.slice(last))}</span>`);
    }
    return out.join("");
  }

  async function copyToClipboard(text) {
    try { await navigator.clipboard.writeText(text); return true; }
    catch (_) {
      const ta = document.createElement("textarea");
      ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
      document.body.appendChild(ta); ta.select();
      let ok = false;
      try { ok = document.execCommand("copy"); } catch (_) {}
      document.body.removeChild(ta);
      return ok;
    }
  }

  function flashCopied(btn) {
    const o = btn.textContent;
    btn.textContent = "✓ Copied";
    btn.classList.add("copied");
    setTimeout(() => { btn.textContent = o; btn.classList.remove("copied"); }, 1200);
  }

  function scrollToBottom() {
    requestAnimationFrame(() => {
      $stream.scrollTop = $stream.scrollHeight;
    });
  }

  // ---- Section rendering ----

  function appendSection(index, role, model) {
    // Finalize the previous section (re-render with code blocks)
    finalizePrevious();

    const sec = document.createElement("div");
    sec.className = `review-section role-${role}`;
    sec.dataset.role = role;
    sec.dataset.model = model;
    sec.innerHTML =
      `<div class="section-header">` +
        `<span class="role-label">Round ${index + 1} · ${role} · ${escapeHtml(model)}</span>` +
        `<button type="button" class="btn ghost copy-section">📋 Copy</button>` +
      `</div>` +
      `<div class="section-body" data-raw=""></div>`;
    $stream.appendChild(sec);
    currentSection = sec;
    scrollToBottom();
  }

  function appendChunkToCurrent(chunk) {
    if (!currentSection) return;
    const body = currentSection.querySelector(".section-body");
    body.dataset.raw = (body.dataset.raw || "") + chunk;
    body.textContent = body.dataset.raw;  // plain text while streaming
    scrollToBottom();
  }

  function finalizePrevious() {
    if (!currentSection) return;
    const body = currentSection.querySelector(".section-body");
    const raw = body.dataset.raw || "";
    body.innerHTML = renderWithCodeBlocks(raw);
  }

  function clearStream() {
    $stream.innerHTML = "";
    currentSection = null;
  }

  // ---- SSE driver ----

  function startReview() {
    const prompt = $promptInput.value.trim();
    if (!prompt) return;

    clearStream();
    $runBtn.disabled = true;
    $stopBtn.classList.remove("hidden");

    fetch("/api/review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        writer_model: $writerModel.value,
        reviewer_model: $reviewerModel.value,
        rounds: parseInt($rounds.value, 10) || 3,
        system_prompt: $systemPrompt.value,
        temperature: parseFloat($temperature.value),
        max_tokens: parseInt($maxTokens.value, 10),
        num_ctx: parseInt($numCtx.value, 10),
      }),
    })
    .then(r => r.json())
    .then(data => {
      if (!data.job_id) throw new Error("no job id returned");
      subscribe(data.job_id);
    })
    .catch(err => {
      $stream.innerHTML = `<div class="review-section role-reviewer"><div class="section-body">Error: ${escapeHtml(String(err))}</div></div>`;
      $runBtn.disabled = false;
      $stopBtn.classList.add("hidden");
    });
  }

  function subscribe(jobId) {
    activeJobId = jobId;
    if (activeSource) activeSource.close();
    const src = new EventSource(`/api/jobs/${jobId}/stream`);
    activeSource = src;

    src.addEventListener("section_start", (e) => {
      try {
        const d = JSON.parse(e.data);
        appendSection(d.index, d.role, d.model);
      } catch (err) { console.warn("section_start parse", err); }
    });

    src.addEventListener("token", (e) => {
      try {
        const d = JSON.parse(e.data);
        if (d.chunk) appendChunkToCurrent(d.chunk);
      } catch (err) { console.warn("token parse", err); }
    });

    src.addEventListener("done", (e) => {
      try {
        finalizePrevious();
        // The done payload has the structured sections — could re-render
        // from those for clean state, but we already streamed them in.
      } finally {
        src.close();
        activeSource = null;
        activeJobId = null;
        $runBtn.disabled = false;
        $stopBtn.classList.add("hidden");
      }
    });

    src.onerror = () => console.warn("EventSource error; will retry");
  }

  // ---- Wire up controls ----

  $runBtn.addEventListener("click", startReview);
  $stopBtn.addEventListener("click", async () => {
    if (!activeJobId) return;
    await fetch(`/api/jobs/${activeJobId}/stop`, { method: "POST" });
  });

  // Copy buttons (delegated)
  $stream.addEventListener("click", async (e) => {
    const t = e.target;
    if (t.classList.contains("copy-section")) {
      const raw = t.closest(".review-section").querySelector(".section-body").dataset.raw || "";
      const ok = await copyToClipboard(raw);
      if (ok) flashCopied(t);
    } else if (t.classList.contains("copy-code")) {
      const code = t.closest(".code-block").querySelector("pre code").textContent;
      const ok = await copyToClipboard(code);
      if (ok) flashCopied(t);
    }
  });

  // Ctrl/Cmd+Enter in the prompt to run.
  $promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      startReview();
    }
  });
})();
