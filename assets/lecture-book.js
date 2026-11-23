const controls = document.querySelector(".book-tools");
const solutions = Array.from(document.querySelectorAll(".book-part details:not(.plot-data):not(.live-data)"));
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
