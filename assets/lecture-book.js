const controls = document.querySelector(".book-tools");
const solutions = Array.from(document.querySelectorAll(".book-part details:not(.plot-data):not(.live-data):not(.study-aside)"));
const toggle = controls.querySelector("[data-solutions]");
function updateLabel() {
  toggle.textContent = solutions.every(solution => solution.open)
    ? "Hide all solutions" : "Show all solutions";
}
toggle.addEventListener("click", () => {
  const open = !solutions.every(solution => solution.open);
  for (const solution of solutions) solution.open = open;
  updateLabel();
});
for (const solution of solutions) solution.addEventListener("toggle", updateLabel);
controls.querySelector("[data-print]").addEventListener("click", () => window.print());
controls.hidden = false;

let collapsedForReading = [];
window.addEventListener("beforeprint", () => {
  for (const panel of document.querySelectorAll("details.study-aside:not([open])")) {
    collapsedForReading.push(panel);
    panel.open = true;
  }
});
window.addEventListener("afterprint", () => {
  for (const panel of collapsedForReading) panel.open = false;
  collapsedForReading = [];
});
