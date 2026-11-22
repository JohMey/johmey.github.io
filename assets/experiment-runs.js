// The table is a complete, readable record of the runs when scripting is unavailable.
for (const table of document.querySelectorAll("table.experiment-runs")) {
  const rows = Array.from(table.tBodies[0].rows);
  const columns = Array.from(table.tHead.rows[0].cells).slice(1);
  const form = document.createElement("form");
  form.className = "experiment-controls";
  form.innerHTML = `
    <fieldset>
      <legend>Compare a change with the baseline</legend>
      <p>Choose different inputs and make a prediction. Compare the saved OneSimulation results below.</p>
      <div class="experiment-fields">
        <label>Changed inputs<select name="scenario"></select></label>
        <label>Quantity<select name="quantity"></select></label>
        <label>Your prediction (optional)<input type="number" name="prediction" step="any"></label>
      </div>
      <div class="experiment-actions">
        <button type="submit" class="button button-primary">Compare runs</button>
        <button type="button" class="button">Show all runs</button>
      </div>
      <output aria-live="polite"></output>
    </fieldset>`;
  const scenario = form.elements.namedItem("scenario");
  const quantity = form.elements.namedItem("quantity");
  const prediction = form.elements.namedItem("prediction");
  const output = form.querySelector("output");
  for (const [index, row] of rows.entries()) {
    if (index > 0) scenario.add(new Option(row.cells[0].textContent.trim(), index));
  }
  for (const [index, header] of columns.entries()) {
    quantity.add(new Option(header.textContent.trim(), index + 1));
  }

  function showBaseline() {
    for (const [index, row] of rows.entries()) row.hidden = index !== 0;
    output.textContent = "";
  }
  scenario.addEventListener("change", () => { prediction.value = ""; showBaseline(); });
  quantity.addEventListener("change", () => { prediction.value = ""; showBaseline(); });
  form.addEventListener("submit", event => {
    event.preventDefault();
    showBaseline();
    const selected = rows[Number(scenario.value)];
    const column = Number(quantity.value);
    selected.hidden = false;
    const lead = prediction.value ? "Your prediction was " + prediction.value + ". " : "";
    output.textContent = lead + columns[column - 1].textContent.trim() + " is "
      + selected.cells[column].textContent.trim() + " for " + selected.cells[0].textContent.trim()
      + " (baseline " + rows[0].cells[column].textContent.trim() + ").";
  });
  form.querySelector('button[type="button"]').addEventListener("click", () => {
    for (const row of rows) row.hidden = false;
    output.textContent = "Showing the baseline and all measured variations.";
  });
  showBaseline();
  const wrapper = document.createElement("div");
  wrapper.className = "experiment-table";
  table.before(form, wrapper);
  wrapper.append(table);
}
