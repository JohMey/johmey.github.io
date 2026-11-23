// The server ships the actual sampled solves. Each parameter choice selects one complete run.
for (const panel of document.querySelectorAll(".model-reader")) {
  const controls = panel.querySelector(".model-inputs");
  const select = controls.querySelector("select");
  const slider = controls.querySelector("input");
  const status = controls.querySelector("output");
  const plot = panel.querySelector("img");
  const download = panel.querySelector("[data-datafile]");
  const label = panel.querySelector("[data-setting]");
  const cache = new Map();
  let rows = [], headers = [], generation = 0;

  function readPoint() {
    const values = rows[Number(slider.value)];
    if (!values) return;
    const format = n => Number(n.toPrecision(6)).toString();
    slider.setAttribute("aria-valuetext", headers[0] + " " + format(values[0]));
    status.textContent = headers.map((h, i) => h + " = " + format(values[i])).join(" · ");
  }

  async function chooseRun() {
    const request = ++generation;
    const value = select.value;
    const file = panel.dataset.filePrefix + value;
    plot.src = file + ".png";
    label.textContent = value;
    download.href = file + ".csv";
    slider.disabled = true;
    status.textContent = "Loading samples…";
    try {
      if (!cache.has(value)) {
        const response = await fetch(download.href);
        if (!response.ok) throw new Error("Unavailable");
        const [header, ...lines] = (await response.text()).trim().split(/\r?\n/);
        cache.set(value, { headers: header.split(","), rows: lines.map(line => line.split(",").map(Number)) });
      }
      if (request !== generation) return;
      const previousFraction = Number(slider.max) ? Number(slider.value) / Number(slider.max) : .5;
      ({ headers, rows } = cache.get(value));
      slider.max = rows.length - 1;
      slider.value = Math.round(previousFraction * (rows.length - 1));
      slider.disabled = false;
      readPoint();
    } catch {
      if (request === generation) status.textContent = "Sample values could not be loaded. Open the CSV to read the run.";
    }
    controls.hidden = false;
  }

  select.addEventListener("change", chooseRun);
  slider.addEventListener("input", readPoint);
  chooseRun();
}
