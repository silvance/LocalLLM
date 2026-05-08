// Compare page: one prompt → multiple models, side by side.
//
// Flow: user picks models + prompt, clicks Run. We POST /api/compare,
// get back N (run_id, jobs[]) and subscribe to each /api/jobs/{id}/stream
// in parallel. On every job's `done` event we capture text + metrics.
// Once all jobs settle, the winner picker + save button appear.
(function () {
  // Copy-pane click delegation — wired before the saved-run early-return
  // so it works in BOTH the saved-run view-only mode (Jinja-rendered
  // buttons) and the live-run mode (JS-rendered buttons). Uses the
  // column's body.dataset.raw — set on `done` for live streams and by
  // the template for loaded runs — so copy works in either mode.
  const $columnsEl = document.getElementById("compare-columns");
  if ($columnsEl) {
    $columnsEl.addEventListener("click", async (e) => {
      const btn = e.target.closest(".copy-pane");
      if (!btn) return;
      const col = btn.closest(".compare-col");
      if (!col) return;
      const body = col.querySelector(".compare-body");
      const text = body.dataset.raw || body.textContent || "";
      try {
        await navigator.clipboard.writeText(text);
        const original = btn.textContent;
        btn.textContent = "✓";
        setTimeout(() => { btn.textContent = original; }, 1100);
      } catch (err) {
        console.warn("copy-pane failed", err);
      }
    });
  }

  const loadedId = window.LOCALLLM && window.LOCALLLM.loadedId;
  if (loadedId) {
    // View-only mode for a saved run — just hydrate raw text.
    document.querySelectorAll(".compare-body").forEach((el) => {
      const raw = el.dataset.raw || el.textContent;
      el.dataset.raw = raw;
      el.textContent = raw;
    });
    wireDeleteButtons();
    return;
  }

  const $columns = document.getElementById("compare-columns");
  const $promptInput = document.getElementById("compare-prompt-input");
  const $systemPrompt = document.getElementById("system-prompt");
  const $runBtn = document.getElementById("run-compare");
  const $status = document.getElementById("compare-status");
  const $afterActions = document.getElementById("compare-actions-after");
  const $winnerPicker = document.getElementById("winner-picker");
  const $saveBtn = document.getElementById("save-compare");

  // Per-run mutable state. Keyed by model_key so multiple jobs don't
  // race over the same column.
  let runState = null;
  let openSources = [];

  function selectedModels() {
    return Array.from(document.querySelectorAll(".model-checkbox"))
      .filter((cb) => cb.checked)
      .map((cb) => cb.value);
  }

  function clearColumns() {
    $columns.innerHTML = "";
  }

  function renderColumn(modelKey) {
    const col = document.createElement("section");
    col.className = "compare-col running";
    col.dataset.modelKey = modelKey;
    col.innerHTML =
      `<header>` +
        `<h3>${modelKey}</h3>` +
        `<code class="model-name">…</code>` +
        `<button type="button" class="btn ghost copy-pane" title="Copy this pane">📋</button>` +
      `</header>` +
      `<div class="compare-meta">⏳ running…</div>` +
      `<div class="compare-body placeholder">Waiting for first token…</div>`;
    $columns.appendChild(col);
    return col;
  }

  function formatMeta(metadata) {
    if (!metadata) return "";
    if (metadata.error) return `❌ ${metadata.error}`;
    const parts = [];
    if (metadata.elapsed_s != null) parts.push(`⏱ ${metadata.elapsed_s.toFixed(1)}s`);
    parts.push(`in ${metadata.prompt_tokens ?? "—"}`);
    parts.push(`out ${metadata.completion_tokens ?? "—"}`);
    if (metadata.tokens_per_sec != null) parts.push(`${metadata.tokens_per_sec.toFixed(1)} tok/s`);
    return parts.join(" · ");
  }

  function subscribe(modelKey, jobId, col) {
    const bodyEl = col.querySelector(".compare-body");
    const metaEl = col.querySelector(".compare-meta");
    const nameEl = col.querySelector(".model-name");
    const src = new EventSource(`/api/jobs/${jobId}/stream`);
    openSources.push(src);

    src.addEventListener("token", (e) => {
      try {
        const data = JSON.parse(e.data);
        if (!data.chunk) return;
        if (bodyEl.classList.contains("placeholder")) {
          bodyEl.textContent = "";
          bodyEl.classList.remove("placeholder");
        }
        bodyEl.textContent += data.chunk;
      } catch {}
    });

    src.addEventListener("done", (e) => {
      try {
        const data = JSON.parse(e.data);
        const metadata = data.metadata || {};
        const text = data.text || bodyEl.textContent;
        bodyEl.textContent = text;
        bodyEl.dataset.raw = text;
        metaEl.textContent = formatMeta(metadata);
        if (metadata.model_name) nameEl.textContent = metadata.model_name;
        col.classList.remove("running");
        col.classList.add(data.status === "error" ? "errored" : "finished");
        runState.results[modelKey] = {
          model_key: modelKey,
          model_name: metadata.model_name || modelKey,
          text,
          prompt_tokens: metadata.prompt_tokens ?? null,
          completion_tokens: metadata.completion_tokens ?? null,
          elapsed_s: metadata.elapsed_s ?? null,
          tokens_per_sec: metadata.tokens_per_sec ?? null,
          error: data.status === "error" ? (data.error || "failed") : null,
        };
        runState.pending -= 1;
        if (runState.pending === 0) onAllDone();
      } finally {
        src.close();
      }
    });

    src.onerror = () => { /* handled by `done` event close */ };
  }

  function onAllDone() {
    $status.textContent = "All models finished.";
    $runBtn.disabled = false;
    // Build the winner radios from the models that succeeded.
    const successful = Object.values(runState.results).filter((r) => !r.error);
    $winnerPicker.innerHTML = `<legend>Pick the winner</legend>`;
    const noneLabel = document.createElement("label");
    noneLabel.innerHTML = `<input type="radio" name="winner" value="" checked> none`;
    $winnerPicker.appendChild(noneLabel);
    for (const r of successful) {
      const lbl = document.createElement("label");
      lbl.innerHTML = `<input type="radio" name="winner" value="${r.model_key}"> 🏆 ${r.model_key}`;
      $winnerPicker.appendChild(lbl);
    }
    $afterActions.classList.remove("hidden");
  }

  // Render columns + subscribe to each per-model job, shared by the
  // fresh-run path (button click) and the resume path (page reload
  // mid-run via window.LOCALLLM.activeRun).
  function attachRun(runId, prompt, system_prompt, jobs) {
    openSources.forEach((s) => s.close());
    openSources = [];
    clearColumns();
    $afterActions && $afterActions.classList.add("hidden");
    runState = {
      runId,
      prompt,
      system_prompt,
      pending: jobs.length,
      results: {},
    };
    $status.textContent = `Running ${jobs.length} model${jobs.length === 1 ? "" : "s"}…`;
    $runBtn.disabled = true;
    for (const job of jobs) {
      const col = renderColumn(job.model_key);
      subscribe(job.model_key, job.job_id, col);
    }
  }

  $runBtn.addEventListener("click", async () => {
    const prompt = $promptInput.value.trim();
    const system_prompt = $systemPrompt.value;
    const models = selectedModels();
    if (!prompt || !models.length) return;

    $status.textContent = "Starting run…";
    $runBtn.disabled = true;

    let res;
    try {
      res = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, system_prompt, models }),
      });
    } catch (err) {
      $status.textContent = `Network error: ${err}`;
      $runBtn.disabled = false;
      return;
    }
    if (!res.ok) {
      const text = await res.text();
      $status.textContent = `Server error ${res.status}: ${text}`;
      $runBtn.disabled = false;
      return;
    }
    const data = await res.json();
    attachRun(data.run_id, prompt, system_prompt, data.jobs);
  });

  // Resume on reload: if the server passed an active run, re-attach
  // SSE streams to the existing jobs and pre-populate the form so the
  // page state matches what the user saw before navigating away.
  const activeRun = window.LOCALLLM && window.LOCALLLM.activeRun;
  if (activeRun) {
    $promptInput.value = activeRun.prompt || "";
    $systemPrompt.value = activeRun.system_prompt || "";
    // Tick the model checkboxes the resumed jobs are running for, so
    // the picker reflects what's actually streaming.
    const activeKeys = new Set(activeRun.jobs.map((j) => j.model_key));
    document.querySelectorAll(".model-checkbox").forEach((cb) => {
      cb.checked = activeKeys.has(cb.value);
    });
    $status.textContent = "Resumed in-progress run.";
    attachRun(activeRun.run_id, activeRun.prompt, activeRun.system_prompt, activeRun.jobs);
  }

  $saveBtn.addEventListener("click", async () => {
    if (!runState) return;
    const winner = (document.querySelector('input[name="winner"]:checked') || {}).value || null;
    const outputs = Object.values(runState.results);
    $saveBtn.disabled = true;
    $saveBtn.textContent = "💾 Saving…";
    try {
      const res = await fetch("/api/compare/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_id: runState.runId,
          prompt: runState.prompt,
          system_prompt: runState.system_prompt,
          outputs,
          winner,
        }),
      });
      if (!res.ok) {
        const text = await res.text();
        $status.textContent = `Save failed: ${text}`;
        $saveBtn.disabled = false;
        $saveBtn.textContent = "💾 Save run";
        return;
      }
      $saveBtn.textContent = "💾 Saved";
      // Refresh the sidebar by reloading — cheaper than rebuilding it
      // from JSON, and the user just confirmed they're done with the run.
      setTimeout(() => { window.location.href = `/compare/${runState.runId}`; }, 400);
    } catch (err) {
      $status.textContent = `Save error: ${err}`;
      $saveBtn.disabled = false;
      $saveBtn.textContent = "💾 Save run";
    }
  });

  wireDeleteButtons();

  function wireDeleteButtons() {
    document.querySelectorAll(".del-compare").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const id = btn.dataset.compareId;
        if (!confirm("Delete this saved comparison?")) return;
        try {
          await fetch(`/compare/${id}`, { method: "DELETE" });
          if (loadedId === id) {
            window.location.href = "/compare";
          } else {
            btn.closest(".chat-row").remove();
          }
        } catch {}
      });
    });
  }
})();
