function applyBenchmarkTolerance(tolerance) {
  document.querySelectorAll("img[data-benchmark]").forEach((img) => {
    const name = img.dataset.benchmark;
    img.src = img.src.replace(/[^/]+$/, `${name}_benchmark_tol_${tolerance}.png`);
    const link = img.closest("a.glightbox");
    if (link) link.href = img.src;
  });
  // Rebuilds glightbox's element list so the zoomed view matches the
  // tolerance just applied, instead of the one baked in at page load
  lightbox.reload();
}

// Keeps the sticky selector flush under Material's fixed header, and pushes
// Material's own "back to top" button down below the selector, instead of
// guessing an offset for either, since the header's height isn't exposed as
// a CSS variable and can shift (e.g. an announcement bar)
function updateToleranceSelectOffset() {
  const header = document.querySelector(".md-header");
  if (!header) return;
  document.documentElement.style.setProperty(
    "--tolerance-select-top",
    `${header.offsetHeight}px`
  );

  const bar = document.querySelector(".tolerance-select");
  const mdTop = document.querySelector(".md-top");
  if (!bar || !mdTop) return;

  // A sticky element's own top stays pinned at the sticky offset once
  // stuck; before that point (page not scrolled far enough yet) it sits
  // lower, in its normal document-flow position
  const stuck = bar.getBoundingClientRect().top <= header.offsetHeight;
  mdTop.style.top = stuck
    ? `${header.offsetHeight + bar.offsetHeight + 12}px`
    : "";
}

document$.subscribe(() => {
  const select = document.getElementById("benchmark-tolerance");
  if (!select) return;
  applyBenchmarkTolerance(select.value);
  select.addEventListener("change", () => applyBenchmarkTolerance(select.value));
  updateToleranceSelectOffset();
});

window.addEventListener("resize", updateToleranceSelectOffset);
window.addEventListener("scroll", updateToleranceSelectOffset, { passive: true });
