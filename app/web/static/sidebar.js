// Sidebar collapse toggle. Shared by all pages with <aside class="sidebar">.
//
// Behavior:
//   * Click the button (#sidebar-toggle) to fold/unfold the sidebar.
//   * State persists in localStorage so the choice survives reloads
//     and reopened tabs.
//   * Initial state is applied BEFORE first paint to avoid a flash of
//     the wrong layout — the early-init block is inlined in each
//     template's <head>; this file only wires the click handler.
(function () {
  const KEY = "localllm.sidebarCollapsed";

  function applyState(collapsed) {
    document.body.classList.toggle("sidebar-collapsed", !!collapsed);
    const btn = document.getElementById("sidebar-toggle");
    if (btn) {
      btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
      btn.title = collapsed ? "Show sidebar" : "Hide sidebar";
    }
  }

  function init() {
    // Sync the body class with whatever the inline early-init wrote
    // so applyState() sees the correct boolean even if localStorage
    // read raced the script load. Cheap to re-evaluate.
    let collapsed;
    try {
      collapsed = localStorage.getItem(KEY) === "1";
    } catch (_) {
      collapsed = false;
    }
    applyState(collapsed);

    const btn = document.getElementById("sidebar-toggle");
    if (!btn) return;
    btn.addEventListener("click", () => {
      const next = !document.body.classList.contains("sidebar-collapsed");
      try { localStorage.setItem(KEY, next ? "1" : "0"); } catch (_) {}
      applyState(next);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
