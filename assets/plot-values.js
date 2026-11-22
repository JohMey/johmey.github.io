// Opening a figure's data view enables keyboard access to each numerical sample.
for (const details of document.querySelectorAll(".plot-data[data-csv]")) {
  details.addEventListener("toggle", async () => {
    if (!details.open || details.dataset.loading) return;
    details.dataset.loading = "true";
    try {
      const response = await fetch(details.dataset.csv);
      if (!response.ok) throw new Error("Data unavailable");
      const lines = (await response.text()).trim().split(/\r?\n/).slice(1);
      const samples = lines.map(line => line.split(",").map(Number));
      const names = [...details.querySelectorAll("thead th")].map(cell => cell.textContent);
      const panel = document.createElement("div");
      panel.className = "plot-reader";
      const label = document.createElement("label");
      label.textContent = "Explore " + names[0];
      const slider = document.createElement("input");
      slider.type = "range";
      slider.min = 0;
      slider.max = samples.length - 1;
      slider.value = Math.floor(samples.length / 2);
      const result = document.createElement("output");
      result.setAttribute("aria-live", "polite");
      const format = new Intl.NumberFormat("en", { maximumSignificantDigits: 5 });
      function update() {
        const values = samples[Number(slider.value)];
        slider.setAttribute("aria-valuetext", names[0] + " " + format.format(values[0]));
        result.textContent = values.map((v, i) => names[i] + " = " + format.format(v)).join(" · ");
      }
      slider.addEventListener("input", update);
      label.append(slider);
      panel.append(label, result);
      details.querySelector(".table-scroll").before(panel);
      update();
    } catch {
      // The static table and download remain available. Reopening retries the request.
      delete details.dataset.loading;
    }
  });
}
