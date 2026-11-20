// A link back to the laboratory's site, first in the header row (before the logo).
document.addEventListener("DOMContentLoaded", () => {
  const header = document.querySelector(".md-header__inner");
  if (!header) return;
  const link = document.createElement("a");
  link.className = "lar-button";
  link.href = "https://lar.upatras.gr/";
  link.textContent = "← LAR";
  link.title = "Laboratory of Automation and Robotics, University of Patras";
  header.prepend(link);
});
