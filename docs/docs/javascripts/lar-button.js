document.addEventListener("DOMContentLoaded", () => {
  const header = document.querySelector(".md-header");
  if (!header) return;
  const link = document.createElement("a");
  link.className = "lar-button";
  link.href = "https://lar.upatras.gr/";
  link.textContent = "← LAR";
  link.title = "Back to the LAR page";
  header.prepend(link);
});
