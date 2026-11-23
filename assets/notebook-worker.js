// Each open notebook has its own local Python process. Only an explicit run starts it.
let runtime;
let namespace;
let activeCell;
let runs = 0;
const base = new URL("/assets/notebook-python/0.28.3/", self.location.origin).href;
const reply = (type, fields = {}) => self.postMessage({type, cell: activeCell, ...fields});
self.notebookFigure = data => reply("figure", {data});

async function initialise(packages) {
  if (runtime) return;
  reply("status", {message: "Loading Python…"});
  importScripts(base + "pyodide.js");
  runtime = await loadPyodide({indexURL: base});
  reply("status", {message: "Loading the scientific libraries…"});
  await runtime.loadPackage(packages);
  runtime.setStdout({batched: text => reply("stream", {name: "stdout", text})});
  runtime.setStderr({batched: text => reply("stream", {name: "stderr", text})});
  await runtime.runPythonAsync(`
import base64 as _base64
import io as _io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt
from js import notebookFigure as _send_figure

def _notebook_show(*args, **kwargs):
    for number in list(_plt.get_fignums()):
        figure = _plt.figure(number)
        buffer = _io.BytesIO()
        figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
        _send_figure(_base64.b64encode(buffer.getvalue()).decode("ascii"))
        _plt.close(figure)

_plt.show = _notebook_show
`);
}

self.onmessage = async ({data}) => {
  try {
    await initialise(data.packages);
    if (!runtime.loadedPackages.scipy && data.cells.some(cell => /\b(?:import|from)\s+scipy\b/.test(cell.code))) {
      reply("status", {message: "Loading SciPy…"});
      await runtime.loadPackage("scipy");
    }
    if (data.restart || !namespace) {
      namespace?.destroy();
      namespace = runtime.runPython("dict(__name__='__main__')");
      runtime.runPython("_plt.close('all')");
      runs = 0;
    }
    for (const cell of data.cells) {
      activeCell = cell.index;
      reply("start", {execution: ++runs});
      let result;
      try {
        result = await runtime.runPythonAsync(cell.code, {globals: namespace});
        if (result !== undefined && result !== null) {
          reply("result", {text: String(result)});
        }
        runtime.runPython("_notebook_show()");
      } finally {
        result?.destroy?.();
      }
    }
    reply("done");
  } catch (err) {
    reply("failed", {message: String(err.message || err)});
  }
};
