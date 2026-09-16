function applyBenchmarkTolerance(tolerance) {
  document.querySelectorAll("img[data-benchmark]").forEach((img) => {
    const name = img.dataset.benchmark;
    img.src = img.src.replace(/[^/]+$/, `${name}_benchmark_tol_${tolerance}.png`);
  });
}

// Keeps the sticky selector flush under Material's fixed header instead of
// a guessed offset, since the header's height isn't exposed as a CSS
// variable and can shift (e.g. an announcement bar)
function updateToleranceSelectOffset() {
  const header = document.querySelector(".md-header");
  if (!header) return;
  document.documentElement.style.setProperty(
    "--tolerance-select-top",
    `${header.offsetHeight}px`
  );
}

document$.subscribe(() => {
  const select = document.getElementById("benchmark-tolerance");
  if (!select) return;
  applyBenchmarkTolerance(select.value);
  select.addEventListener("change", () => applyBenchmarkTolerance(select.value));
  updateToleranceSelectOffset();
});

window.addEventListener("resize", updateToleranceSelectOffset);
