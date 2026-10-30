/* ==============================================================
   CreditSense AI — Shared client-side utilities
   Loaded on every page. Exposes window.CS with small helpers that
   individual page scripts (assessment, batch, dashboards) build on.
   ============================================================== */

(function () {
  "use strict";

  /* ---------------- THEME ---------------- */
  const body = document.body;
  const themeToggle = document.getElementById("themeToggle");
  const themeLabel = document.getElementById("themeLabel");

  function applyTheme(theme) {
    body.setAttribute("data-theme", theme);
    if (themeToggle) themeToggle.setAttribute("aria-pressed", theme === "dark");
    if (themeLabel) themeLabel.textContent = theme === "dark" ? "Light mode" : "Dark mode";
    try { localStorage.setItem("creditsense_theme", theme); } catch (e) {}
  }

  (function initTheme() {
    let saved = null;
    try { saved = localStorage.getItem("creditsense_theme"); } catch (e) {}
    if (!saved) {
      saved = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    }
    applyTheme(saved);
  })();

  if (themeToggle) {
    themeToggle.addEventListener("click", function () {
      applyTheme(body.getAttribute("data-theme") === "dark" ? "light" : "dark");
    });
  }

  /* ---------------- ACTIVE NAV LINK ---------------- */
  (function highlightActiveNav() {
    const path = window.location.pathname;
    document.querySelectorAll(".top-nav a").forEach(function (link) {
      const href = link.getAttribute("href");
      if (href === path || (href !== "/" && path.startsWith(href))) {
        link.classList.add("active");
      }
    });
  })();

  /* ---------------- TOAST ---------------- */
  let toastEl = document.getElementById("toast");
  if (!toastEl) {
    toastEl = document.createElement("div");
    toastEl.id = "toast";
    toastEl.className = "toast";
    toastEl.setAttribute("role", "status");
    document.body.appendChild(toastEl);
  }
  let toastTimer = null;
  function showToast(message, isError) {
    toastEl.textContent = message;
    toastEl.classList.toggle("error", !!isError);
    toastEl.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { toastEl.classList.remove("show"); }, 3200);
  }

  /* ---------------- CIBIL SCORE HELPERS (shared across pages) ---------------- */
  function cibilBand(score) {
    if (score >= 750) return { label: "Excellent", color: "#15803D" };
    if (score >= 650) return { label: "Good", color: "#2563EB" };
    if (score >= 550) return { label: "Fair", color: "#B45309" };
    return { label: "Poor", color: "#B91C1C" };
  }

  /* ---------------- FOOTER YEAR ---------------- */
  const yearEl = document.getElementById("year");
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  /* ---------------- EXPORT ---------------- */
  window.CS = {
    showToast: showToast,
    cibilBand: cibilBand,
  };
})();
