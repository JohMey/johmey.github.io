// All data are computed in this page. A chapter selects its own registered physical model.
(() => {
  const colors = ["#174d59", "#ad491d", "#695082", "#64721e", "#2d72ab", "#97446a"];
  const format = value => {
    if (!Number.isFinite(value)) return "—";
    const size = Math.abs(value);
    return size && (size < 0.001 || size >= 100000) ? value.toExponential(2) : String(Number(value.toPrecision(5)));
  };
  function element(tag, parent, content) {
    const el = document.createElement(tag);
    if (content !== undefined) el.textContent = content;
    parent.append(el);
    return el;
  }

  function draw(canvas, data, xLabel, yLabel) {
    const width = Math.max(260, Math.round(canvas.clientWidth || 650));
    const height = width < 420 ? 245 : 300;
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    canvas.style.height = height + "px";
    const ctx = canvas.getContext("2d");
    ctx.scale(ratio, ratio);
    const left = width < 420 ? 60 : 72, right = width - 20, top = 16, bottom = height - 51;
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const series of data.curves) {
      for (const [x,y] of series.points) {
        if (Number.isFinite(x) && Number.isFinite(y)) {
          xmin = Math.min(xmin, x); xmax = Math.max(xmax, x);
          ymin = Math.min(ymin, y); ymax = Math.max(ymax, y);
        }
      }
    }
    if (!Number.isFinite(xmin)) return;
    if (xmin === xmax) { xmin -= .5; xmax += .5; }
    if (ymin === ymax) { const pad = Math.max(.05, Math.abs(ymax) * .04); ymin -= pad; ymax += pad; }
    const extra = (ymax - ymin) * .045; ymin -= extra; ymax += extra;
    const X = x => left + (x-xmin)/(xmax-xmin)*(right-left);
    const Y = y => bottom - (y-ymin)/(ymax-ymin)*(bottom-top);
    ctx.font = "11px system-ui, sans-serif";
    ctx.strokeStyle = "#d8dedc"; ctx.fillStyle = "#283a3c"; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const x = xmin + (xmax-xmin)*i/4, y = ymin + (ymax-ymin)*i/4;
      ctx.beginPath(); ctx.moveTo(X(x),top); ctx.lineTo(X(x),bottom);
      ctx.moveTo(left,Y(y)); ctx.lineTo(right,Y(y)); ctx.stroke();
      ctx.textAlign = i === 4 ? "right" : i === 0 ? "left" : "center"; ctx.fillText(format(x),X(x),bottom+17);
      ctx.textAlign = "right"; ctx.fillText(format(y),left-6,Y(y)+4);
    }
    ctx.textAlign = "center"; ctx.font = "13px system-ui, sans-serif";
    ctx.fillText(xLabel,(left+right)/2,height-5,right-left+22);
    ctx.save(); ctx.translate(12,(top+bottom)/2); ctx.rotate(-Math.PI/2);
    ctx.fillText(yLabel,0,0,height-25); ctx.restore();
    data.curves.forEach((series,j) => {
      ctx.strokeStyle = ctx.fillStyle = colors[j%colors.length]; ctx.lineWidth = 2.2;
      ctx.setLineDash([[],[7,3],[2,3],[9,3,2,3]][j%4]);
      ctx.beginPath(); let started = false;
      for (const [x,y] of series.points) {
        if (!Number.isFinite(x) || !Number.isFinite(y)) { started = false; continue; }
        if (!started) ctx.moveTo(X(x),Y(y)); else ctx.lineTo(X(x),Y(y));
        started = true;
      }
      ctx.stroke(); ctx.setLineDash([]);
      if (series.points.length <= 20) for (const [x,y] of series.points) {
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        ctx.beginPath(); ctx.arc(X(x),Y(y),2.8,0,2*Math.PI); ctx.fill();
      }
    });
  }

  function initialise(panel) {
    const model = window.fundamentalStudies?.[panel.dataset.study];
    if (!model) return;
    const form = element("fieldset",panel);
    element("legend",form,"Try the calculation");
    const controls = element("div",form); controls.className = "live-inputs";
    const values = model.controls.map(([, , , ,initial]) => initial);
    const inputs = model.controls.map(([name,min,max,step,initial], i) => {
      const label = element("label",controls);
      const head = element("span",label,name + " ");
      const value = element("b",head,format(initial));
      const input = element("input",label);
      input.type = "range"; input.min = min; input.max = max; input.step = step; input.value = initial;
      input.setAttribute("aria-label",name);
      input.addEventListener("input",() => { values[i] = Number(input.value); value.textContent = format(values[i]); update(); });
      return {input,value};
    });
    const figure = element("figure",form);
    const canvas = element("canvas",figure); canvas.setAttribute("role","img");
    const legend = element("figcaption",figure);
    const result = element("output",form); result.setAttribute("aria-live","polite");
    let command;
    if (model.localModel) {
      const local = element("div", form);
      local.className = "live-reproduce";
      const download = element("a", local, "Download the OneSimulation model");
      download.href = "/downloads/fundamentals/" + model.localModel.file;
      download.download = model.localModel.file;
      element("p", local, "In your OneSimulation environment, run the model with these inputs.");
      command = element("code", local);
    }
    const actions = element("div",form); actions.className = "live-actions";
    const reset = element("button",actions,"Reset inputs"); reset.type = "button";
    const dataView = element("details",form); dataView.className="live-data";
    element("summary",dataView,"Read numerical values");
    const tableWrap = element("div",dataView); tableWrap.className="table-wrap";
    let data;
    function showTable() {
      tableWrap.replaceChildren();
      if (!dataView.open) return;
      const t = element("table",tableWrap);
      const row = element("tr",element("thead",t));
      for (const s of ["Curve",model.xLabel,model.yLabel]) element("th",row,s).scope="col";
      const body = element("tbody",t);
      for (const s of data.curves) {
        const points=s.points.filter(p=>p.every(Number.isFinite));
        const selected = [...new Set(Array.from({length:Math.min(7,points.length)},(_,i)=>Math.round(i*(points.length-1)/Math.max(1,Math.min(7,points.length)-1))))];
        for(const j of selected) {
          const tr=element("tr",body); element("th",tr,s.name).scope="row";
          element("td",tr,format(points[j][0])); element("td",tr,format(points[j][1]));
        }
      }
    }
    function plot(){ if(data) draw(canvas,data,model.xLabel,model.yLabel); }
    function update() {
      data = model.calculate(values.slice());
      result.textContent = data.result;
      if (command) command.textContent = "python " + model.localModel.file + " " + model.localModel.args(values);
      legend.replaceChildren();
      data.curves.forEach((s,i) => {
        const tag=element("span",legend,s.name); tag.style.borderColor=colors[i%colors.length];
      });
      canvas.setAttribute("aria-label",model.yLabel+" against "+model.xLabel+". "+data.result);
      showTable(); plot();
    }
    reset.addEventListener("click",()=> {
      model.controls.forEach((spec,i)=>{values[i]=spec[4];inputs[i].input.value=spec[4];inputs[i].value.textContent=format(spec[4]);}); update();
    });
    dataView.addEventListener("toggle",showTable);
    const observer = new ResizeObserver(entries=>{if(entries[0].contentRect.width !== canvas.lastPlotWidth){canvas.lastPlotWidth=entries[0].contentRect.width;plot();}});
    observer.observe(figure);
    update();
  }
  for (const panel of document.querySelectorAll(".interactive-study[data-study]")) initialise(panel);
})();
