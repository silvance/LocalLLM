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
  const $fallbackWriterModel = document.getElementById("fallback-writer-model");
  const $rounds = document.getElementById("rounds");
  const $applyPriorLessons = document.getElementById("apply-prior-lessons");
  const $systemPrompt = document.getElementById("system-prompt");
  const $temperature = document.getElementById("temperature");
  const $maxTokens = document.getElementById("max-tokens");
  const $numCtx = document.getElementById("num-ctx");

  let activeJobId = null;
  let activeSource = null;
  let currentSection = null;  // DOM element for the in-progress section
  let lastSectionIndex = -1;  // Highest section_start index we've rendered (dedup)

  // ---- per-tab settings persistence ----
  // The fallback-writer dropdown silently reset to "(none)" whenever the
  // operator tabbed away and came back to /review (the template renders a
  // fresh page, and there's no `loaded` row when the URL has no review id).
  // Cache the user's choice client-side so a tab-off doesn't lose it. The
  // server-side ReviewSession also persists fallback_writer_model now —
  // this localStorage path is the belt-and-suspenders for the
  // before-the-review-is-submitted state.
  const LS_KEY_FALLBACK = "localllm.review.fallbackWriterModel";

  function rememberFallback() {
    if (!$fallbackWriterModel) return;
    try {
      localStorage.setItem(LS_KEY_FALLBACK, $fallbackWriterModel.value || "");
    } catch (_) { /* private mode / quota → ignore, no-op fallback */ }
  }

  function restoreFallback() {
    if (!$fallbackWriterModel) return;
    // Server-rendered `loaded.fallback_writer_model` wins — that's the
    // authoritative value for an in-flight review. Only consult
    // localStorage when the page is fresh (no loaded review).
    const loadedVal = $fallbackWriterModel.dataset.loadedValue || "";
    if (loadedVal) return;
    try {
      const saved = localStorage.getItem(LS_KEY_FALLBACK);
      if (saved == null) return;
      // Only restore if the saved option actually exists in the
      // dropdown — model lists shift between sessions.
      const opt = Array.from($fallbackWriterModel.options)
        .find(o => o.value === saved);
      if (opt) $fallbackWriterModel.value = saved;
    } catch (_) { /* ignore */ }
  }

  if ($fallbackWriterModel) {
    restoreFallback();
    $fallbackWriterModel.addEventListener("change", rememberFallback);
  }

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

  // ---- Status banner (visible without devtools) ----

  function setStatusBanner(msg, isError = false) {
    let banner = document.getElementById("review-status-banner");
    if (!banner) {
      banner = document.createElement("div");
      banner.id = "review-status-banner";
      banner.style.cssText = "padding:0.4rem 0.85rem;font-size:0.85rem;border-radius:6px;margin:0.4rem 0;";
      const promptArea = document.querySelector(".review-prompt");
      if (promptArea) promptArea.appendChild(banner);
    }
    if (!msg) {
      banner.style.display = "none";
      banner.textContent = "";
      return;
    }
    banner.style.display = "block";
    banner.textContent = msg;
    banner.style.background = isError ? "rgba(248,81,73,0.18)" : "rgba(47,129,247,0.15)";
    banner.style.color = isError ? "#f5b8b3" : "var(--text)";
  }

  function flashCopied(btn) {
    const o = btn.textContent;
    btn.textContent = "✓ Copied";
    btn.classList.add("copied");
    setTimeout(() => { btn.textContent = o; btn.classList.remove("copied"); }, 1200);
  }

  // ---- Lint panel (live pyflakes findings keyed by section index) ----

  function updateLintPanel(sectionIdx, findings) {
    const sec = $stream.querySelector(
      `.review-section[data-section-index="${sectionIdx}"]`
    );
    if (!sec) return;

    let panel = sec.querySelector(".lint-panel");
    if (!panel) {
      panel = document.createElement("div");
      panel.className = "lint-panel";
      sec.appendChild(panel);
    }

    if (!findings || findings.length === 0) {
      panel.innerHTML = '<div class="lint-summary clean">✅ Lint: clean (no issues)</div>';
      return;
    }

    let errors = 0, warnings = 0;
    for (const f of findings) {
      if (f.severity === "error") errors++;
      else warnings++;
    }
    const summary = errors > 0
      ? `❌ Lint: ${errors} error${errors !== 1 ? "s" : ""}, ${warnings} warning${warnings !== 1 ? "s" : ""}`
      : `⚠️ Lint: ${warnings} warning${warnings !== 1 ? "s" : ""}`;
    const summaryClass = errors > 0 ? "has-errors" : "has-warnings";

    const items = findings.map(f => {
      const loc = `B${(f.block ?? 0) + 1}:L${f.line}`;
      return `<li class="lint-item ${escapeHtml(f.severity)}">` +
               `<span class="lint-loc">${escapeHtml(loc)}</span>` +
               `<span class="lint-msg">${escapeHtml(f.message)}</span>` +
             `</li>`;
    }).join("");

    panel.innerHTML =
      `<details open>` +
        `<summary class="lint-summary ${summaryClass}">${summary}</summary>` +
        `<ul class="lint-list">${items}</ul>` +
      `</details>`;
  }

  // ---- Line diff (LCS-based, ~O(m*n) — fine for typical code length) ----

  function computeLineDiff(oldText, newText) {
    const a = oldText.split("\n");
    const b = newText.split("\n");
    const m = a.length;
    const n = b.length;
    // dp[i][j] = LCS length of a[..i] and b[..j]
    const dp = new Array(m + 1);
    for (let i = 0; i <= m; i++) dp[i] = new Int32Array(n + 1);
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        if (a[i - 1] === b[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1;
        else dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
    // Walk backwards through the table to build the diff sequence.
    const out = [];
    let i = m, j = n;
    while (i > 0 || j > 0) {
      if (i > 0 && j > 0 && a[i - 1] === b[j - 1]) {
        out.push({ type: "same", text: a[i - 1] });
        i--; j--;
      } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
        out.push({ type: "add", text: b[j - 1] });
        j--;
      } else {
        out.push({ type: "del", text: a[i - 1] });
        i--;
      }
    }
    return out.reverse();
  }

  function renderDiffHtml(lines) {
    let added = 0, removed = 0;
    const rows = [];
    for (const line of lines) {
      if (line.type === "add") added++;
      else if (line.type === "del") removed++;
      const marker = line.type === "add" ? "+" : line.type === "del" ? "−" : " ";
      const text = line.text === "" ? " " : line.text;
      rows.push(
        `<div class="diff-line ${line.type}">` +
          `<span class="diff-marker">${marker}</span>` +
          `<span class="diff-text">${escapeHtml(text)}</span>` +
        `</div>`
      );
    }
    return (
      `<div class="diff-summary">` +
        `<span class="add-count">+${added}</span> ` +
        `<span class="del-count">−${removed}</span>` +
      `</div>` +
      `<div class="diff-view">${rows.join("")}</div>`
    );
  }

  function findPreviousWriterSection(sectionEl) {
    let prev = sectionEl.previousElementSibling;
    while (prev) {
      if (prev.classList && prev.dataset && prev.dataset.role === "writer") return prev;
      prev = prev.previousElementSibling;
    }
    return null;
  }

  function maybeAddDiffToggle(sectionEl) {
    if (!sectionEl || sectionEl.dataset.role !== "writer") return;
    if (sectionEl.querySelector(".toggle-diff")) return;  // idempotent
    if (!findPreviousWriterSection(sectionEl)) return;    // no peer to diff against

    const header = sectionEl.querySelector(".section-header");
    if (!header) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn ghost toggle-diff";
    btn.textContent = "🔀 Show diff";
    // Place the diff toggle just before the copy button.
    const copyBtn = header.querySelector(".copy-section");
    if (copyBtn) {
      header.insertBefore(btn, copyBtn);
    } else {
      header.appendChild(btn);
    }
  }

  function toggleDiff(sectionEl) {
    const body = sectionEl.querySelector(".section-body");
    const btn = sectionEl.querySelector(".toggle-diff");
    if (!body || !btn) return;

    if (sectionEl.dataset.viewMode === "diff") {
      // Switch back to rendered code
      body.innerHTML = renderWithCodeBlocks(body.dataset.raw || "");
      sectionEl.dataset.viewMode = "code";
      btn.textContent = "🔀 Show diff";
      return;
    }
    const prev = findPreviousWriterSection(sectionEl);
    if (!prev) return;
    const prevBody = prev.querySelector(".section-body");
    const oldText = (prevBody && prevBody.dataset.raw) || "";
    const newText = body.dataset.raw || "";
    const diff = computeLineDiff(oldText, newText);
    body.innerHTML = renderDiffHtml(diff);
    sectionEl.dataset.viewMode = "diff";
    btn.textContent = "📄 Show code";
  }

  function scrollToBottom() {
    requestAnimationFrame(() => {
      $stream.scrollTop = $stream.scrollHeight;
    });
  }

  function isScrolledNearBottom() {
    return $stream.scrollHeight - $stream.scrollTop - $stream.clientHeight < 120;
  }

  // ---- Section rendering ----

  function appendSection(index, role, model) {
    // Skip duplicates from the rare race where SSE replay + live queue both
    // contain the same section_start.
    if (typeof index === "number" && index <= lastSectionIndex) return;
    if (typeof index === "number") lastSectionIndex = index;

    // Finalize the previous section (re-render with code blocks)
    finalizePrevious();

    const sec = document.createElement("details");
    sec.open = true;
    sec.className = `review-section role-${role}`;
    sec.dataset.role = role;
    sec.dataset.model = model;
    sec.dataset.sectionIndex = String(index);
    sec.dataset.startedAt = String(Date.now());
    sec.dataset.tokenCount = "0";
    sec.innerHTML =
      `<summary class="section-header">` +
        `<span class="role-label">Round ${index + 1} · ${role} · ${escapeHtml(model)}</span>` +
        `<span class="section-stats" data-stats="">…</span>` +
        `<button type="button" class="btn ghost copy-section">📋 Copy</button>` +
      `</summary>` +
      `<div class="section-body section-loading" data-raw="">⏳ Loading ${escapeHtml(role)} (${escapeHtml(model)})…</div>`;
    $stream.appendChild(sec);
    currentSection = sec;
    scrollToBottom();
  }

  function appendChunkToCurrent(chunk) {
    if (!currentSection) return;
    const body = currentSection.querySelector(".section-body");
    if (body.classList.contains("section-loading")) {
      body.classList.remove("section-loading");
      body.textContent = "";
    }
    body.dataset.raw = (body.dataset.raw || "") + chunk;
    body.textContent = body.dataset.raw;  // plain text while streaming

    // Live "X tokens" counter — rough word-count proxy. Better than nothing
    // and gives the user something to look at while a slow model warms up.
    const tokens = (body.dataset.raw.match(/\S+/g) || []).length;
    currentSection.dataset.tokenCount = String(tokens);
    const stats = currentSection.querySelector(".section-stats");
    if (stats) stats.textContent = `${tokens.toLocaleString()} tokens…`;

    // Only auto-scroll if the user was already near the bottom — don't yank
    // them away from older sections they're trying to read.
    if (isScrolledNearBottom()) scrollToBottom();
  }

  function finalizePrevious() {
    if (!currentSection) return;
    const body = currentSection.querySelector(".section-body");
    const raw = body.dataset.raw || "";

    // Replace with rendered markdown OR an explicit empty marker so the
    // user can see which sections produced no output.
    if (!raw.trim()) {
      body.classList.remove("section-loading");
      body.innerHTML = '<span class="empty-marker">⚠️ (model returned no output for this section)</span>';
    } else {
      body.classList.remove("section-loading");
      body.innerHTML = renderWithCodeBlocks(raw);
    }
    currentSection.dataset.viewMode = "code";

    // Final stats: token count + elapsed seconds.
    const tokens = parseInt(currentSection.dataset.tokenCount || "0", 10);
    const startedAt = parseInt(currentSection.dataset.startedAt || "0", 10);
    const elapsed = startedAt ? ((Date.now() - startedAt) / 1000).toFixed(1) : null;
    const stats = currentSection.querySelector(".section-stats");
    if (stats) {
      const parts = [];
      parts.push(`${tokens.toLocaleString()} tokens`);
      if (elapsed) parts.push(`${elapsed}s`);
      if (tokens > 0 && elapsed && parseFloat(elapsed) > 0) {
        parts.push(`${(tokens / parseFloat(elapsed)).toFixed(1)} tok/s`);
      }
      stats.textContent = parts.join(" · ");
    }

    // Add the diff toggle on writer revisions (any writer section that has
    // a previous writer section to diff against).
    maybeAddDiffToggle(currentSection);
  }

  function clearStream() {
    $stream.innerHTML = "";
    currentSection = null;
    lastSectionIndex = -1;
  }

  // ---- SSE driver ----

  function startReview() {
    try {
      _startReviewImpl();
    } catch (err) {
      console.error("startReview threw", err);
      const msg = `Could not start review: ${err && err.message ? err.message : String(err)}`;
      $stream.innerHTML = `<div class="review-section role-reviewer"><div class="section-body">${escapeHtml(msg)}</div></div>`;
      setStatusBanner(msg, true);
      $runBtn.disabled = false;
      $stopBtn.classList.add("hidden");
    }
  }

  function _startReviewImpl() {
    const prompt = $promptInput.value.trim();
    if (!prompt) {
      setStatusBanner("Type a prompt first.", true);
      return;
    }
    setStatusBanner("");

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
        fallback_writer_model: $fallbackWriterModel ? $fallbackWriterModel.value : "",
        rounds: parseInt($rounds.value, 10) || 3,
        apply_prior_lessons: !!($applyPriorLessons && $applyPriorLessons.checked),
        system_prompt: $systemPrompt.value,
        temperature: parseFloat($temperature.value),
        max_tokens: parseInt($maxTokens.value, 10),
        num_ctx: parseInt($numCtx.value, 10),
      }),
    })
    .then(r => {
      if (!r.ok) return r.text().then(t => { throw new Error(`POST /api/review ${r.status}: ${t}`); });
      return r.json();
    })
    .then(data => {
      if (!data.job_id) throw new Error("server returned no job_id");
      subscribe(data.job_id);
    })
    .catch(err => {
      console.error("review start failed", err);
      const msg = err && err.message ? err.message : String(err);
      $stream.innerHTML = `<div class="review-section role-reviewer"><div class="section-body">${escapeHtml(msg)}</div></div>`;
      setStatusBanner(msg, true);
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

    src.addEventListener("lint_state", (e) => {
      try {
        const d = JSON.parse(e.data);
        updateLintPanel(d.section_index, d.findings || []);
      } catch (err) { console.warn("lint_state parse", err); }
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

  // Copy buttons (delegated). Buttons live inside <summary> elements
  // (the section is now a <details> for collapse), so any click that
  // belongs to a real button must preventDefault — otherwise the click
  // bubbles to summary and toggles the section closed.
  $stream.addEventListener("click", async (e) => {
    const t = e.target;
    if (t.classList.contains("copy-section")) {
      e.preventDefault();
      const raw = t.closest(".review-section").querySelector(".section-body").dataset.raw || "";
      const ok = await copyToClipboard(raw);
      if (ok) flashCopied(t);
    } else if (t.classList.contains("copy-code")) {
      e.preventDefault();
      const code = t.closest(".code-block").querySelector("pre code").textContent;
      const ok = await copyToClipboard(code);
      if (ok) flashCopied(t);
    } else if (t.classList.contains("toggle-diff")) {
      e.preventDefault();
      toggleDiff(t.closest(".review-section"));
    }
  });

  // Ctrl/Cmd+Enter in the prompt to run.
  $promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      startReview();
    }
  });

  // Delete a saved review from the sidebar list.
  document.querySelectorAll(".del-review").forEach(btn => {
    btn.addEventListener("click", async () => {
      const rid = btn.dataset.reviewId;
      if (!confirm("Delete this review?")) return;
      try {
        await fetch(`/review/${rid}`, { method: "DELETE" });
        // If we were viewing it, jump back to /review (clean slate)
        if (window.location.pathname.endsWith(`/review/${rid}`)) {
          window.location.href = "/review";
        } else {
          btn.closest(".chat-row").remove();
        }
      } catch (err) {
        console.warn("delete failed", err);
      }
    });
  });

  // If we're viewing a saved review (sections rendered server-side), hydrate
  // them with code-block rendering and don't try to start a new generation.
  function hydrateSavedSections() {
    document.querySelectorAll("#review-stream .review-section").forEach(sec => {
      const body = sec.querySelector(".section-body");
      const raw = body.dataset.raw || body.textContent;
      body.dataset.raw = raw;
      body.innerHTML = renderWithCodeBlocks(raw);
      sec.dataset.viewMode = "code";
      maybeAddDiffToggle(sec);
    });
  }

  hydrateSavedSections();

  // Resume on reload: server passed an in-flight review job_id, so
  // re-attach the SSE stream. The /api/jobs/{id}/stream endpoint
  // replays buffered text + checkpoints so section_start markers and
  // partial sections render correctly even though we joined late.
  const initialJobId = window.LOCALLLM && window.LOCALLLM.activeJobId;
  if (initialJobId) {
    subscribe(initialJobId);
  }

  // Visible "I am alive" probe so a broken JS init doesn't look the same
  // as "user clicked but nothing happened".
  try {
    if ($runBtn) {
      $runBtn.dataset.jsReady = "1";
      console.info("[review] JS initialized, runBtn =", $runBtn);
    }
  } catch (_) { /* never break init on logging */ }
})();
