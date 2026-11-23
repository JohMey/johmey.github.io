(() => {
  const notebook = document.querySelector("[data-notebook-editor]");
  if (!notebook || !window.Worker) return;
  const status = notebook.querySelector("[data-kernel-status]");
  const all = notebook.querySelector("[data-run-all]");
  const stop = notebook.querySelector("[data-stop]");
  const cells = [...notebook.querySelectorAll("[data-code-cell]")];
  const save = notebook.querySelector("[data-save-notebook]");
  const packages = JSON.parse(notebook.dataset.packages);
  let worker;
  let busy = false;
  let active;
  const outputs = cells.map(() => []);
  const execution = cells.map(() => null);

  function setBusy(state) {
    busy = state;
    all.disabled = state;
    stop.disabled = !state;
    for (const cell of cells) {
      cell.querySelector("button").disabled = state;
      cell.querySelector("textarea").readOnly = state;
    }
  }
  function resetCell(index) {
    cells[index].querySelector(".notebook-output").replaceChildren();
    cells[index].querySelector(".cell-execution").textContent = "";
    outputs[index] = [];
    execution[index] = null;
  }
  function showText(index, text, kind) {
    const output = cells[index].querySelector(".notebook-output");
    const last = output.lastElementChild;
    if (kind === "stdout" && last?.dataset.kind === kind) {
      last.textContent += "\n" + text;
    } else {
      const pre = document.createElement("pre");
      pre.dataset.kind = kind;
      pre.textContent = text;
      output.append(pre);
    }
  }
  function receive({data}) {
    const i = data.cell;
    switch (data.type) {
      case "status":
        status.textContent = data.message;
        break;
      case "start":
        active = i;
        resetCell(i);
        execution[i] = data.execution;
        cells[i].querySelector(".cell-execution").textContent = "[" + data.execution + "]";
        status.textContent = "Running cell " + (i + 1) + " of " + cells.length + "…";
        break;
      case "stream":
        showText(i, data.text, data.name);
        outputs[i].push({output_type:"stream", name:data.name, text:data.text+"\n"});
        break;
      case "result":
        showText(i, data.text, "result");
        outputs[i].push({output_type:"execute_result", execution_count:execution[i], metadata:{}, data:{"text/plain":data.text}});
        break;
      case "figure": {
        const img = document.createElement("img");
        img.src = "data:image/png;base64," + data.data;
        img.alt = "Calculated figure from cell " + (i + 1);
        cells[i].querySelector(".notebook-output").append(img);
        outputs[i].push({output_type:"display_data", metadata:{}, data:{"image/png":data.data}});
        break;
      }
      case "done":
        setBusy(false);
        status.textContent = "Run complete. Change an input and run again.";
        break;
      case "failed":
        setBusy(false);
        if (Number.isInteger(active)) {
          showText(active, data.message, "error");
          outputs[active].push({output_type:"error", ename:"PythonError", evalue:data.message, traceback:data.message.split("\n")});
          status.textContent = "Execution stopped at cell " + (active + 1) + ".";
        } else {
          status.textContent = "Python could not start. Retry or download the notebook for Jupyter.";
          worker?.terminate();
          worker = undefined;
        }
        break;
    }
  }
  function run(indices, restart) {
    if (busy) return;
    if (!worker) {
      worker = new Worker("/assets/notebook-worker.js");
      worker.onmessage = receive;
      worker.onerror = () => {
        worker?.terminate(); worker = undefined; setBusy(false);
        status.textContent = "The Python process stopped. Run all cells to restart it.";
      };
    }
    active = undefined;
    if (restart) cells.forEach((_, i) => resetCell(i));
    setBusy(true);
    status.textContent = "Starting the calculation…";
    worker.postMessage({restart, packages, cells:indices.map(i=>({index:i,code:cells[i].querySelector("textarea").value}))});
  }
  all.addEventListener("click",()=>run(cells.map((_,i)=>i),true));
  stop.addEventListener("click",()=>{
    worker?.terminate(); worker = undefined; setBusy(false);
    status.textContent = "Stopped. Run all cells to reinitialise the variables.";
  });
  cells.forEach((cell, i)=>{
    cell.querySelector("button").addEventListener("click",()=>run([i],false));
    const input = cell.querySelector("textarea");
    input.addEventListener("input", () => {
      if (execution.slice(i).some(count => count !== null)) {
        for (let j = i; j < cells.length; j++) resetCell(j);
      }
      status.textContent = "Input changed. Run all cells to update the results.";
    });
    input.addEventListener("keydown",event=>{
      if (event.shiftKey && event.key==="Enter") {event.preventDefault();run([i],false);}
    });
  });
  save.addEventListener("click",async event=>{
    event.preventDefault();
    try {
      const response = await fetch(save.href);
      if (!response.ok) throw new Error("notebook unavailable");
      const file = await response.json();
      file.cells.filter(c=>c.cell_type==="code").forEach((cell,i)=>{
        cell.source = cells[i].querySelector("textarea").value;
        cell.outputs = outputs[i];
        cell.execution_count = execution[i];
      });
      const url = URL.createObjectURL(new Blob([JSON.stringify(file,null,1)+"\n"],{type:"application/x-ipynb+json"}));
      const link = document.createElement("a");link.href=url;link.download=save.download;link.click();
      setTimeout(()=>URL.revokeObjectURL(url),1000);
    } catch {
      status.textContent = "Could not save edits. The original .ipynb is available from the course notebook list.";
    }
  });
  setBusy(false);
  notebook.querySelector(".notebook-controls").hidden = false;
})();
