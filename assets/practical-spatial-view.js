// Orthographic views of model coordinates. Geometry is supplied by the calculation,
// not a separate illustration; camera changes never change the computed solution.
(() => {
  window.createFundamentalSpatialView = function(parent) {
    const add = (tag, text, host = parent) => {
      const el = document.createElement(tag);
      if (text) el.textContent = text;
      host.append(el); return el;
    };
    const figure = add('figure'); figure.className = 'spatial-view';
    const canvas = add('canvas', '', figure);
    canvas.tabIndex = 0; canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', 'Interactive spatial view. Arrow keys rotate; plus and minus zoom; Home resets the view.');
    const caption = add('figcaption', '', figure);
    const hint = add('p', 'Drag to rotate. Use the arrow keys when the view is focused, or the buttons below.');
    hint.className = 'spatial-hint';
    const buttons = add('div'); buttons.className = 'live-actions spatial-actions';
    let scene, yaw = -.65, pitch = .5, zoom = 1, pointer;
    const clamp = () => { pitch = Math.max(-1.4, Math.min(1.4, pitch)); zoom = Math.max(.5, Math.min(2.5, zoom)); };
    const reset = () => { yaw = -.65; pitch = .5; zoom = 1; draw(); };
    const change = (dy, dp, dz = 1) => { yaw += dy; pitch += dp; zoom *= dz; clamp(); draw(); };
    for (const [label, action] of [
      ['Rotate left', () => change(-.2, 0)], ['Rotate right', () => change(.2, 0)],
      ['Tilt up', () => change(0, .15)], ['Tilt down', () => change(0, -.15)],
      ['Zoom in', () => change(0, 0, 1.2)], ['Zoom out', () => change(0, 0, 1/1.2)],
      ['Reset view', reset],
    ]) {
      const button = add('button', label, buttons); button.type = 'button'; button.addEventListener('click', action);
    }
    canvas.addEventListener('pointerdown', e => {
      if (e.button !== 0) return;
      pointer = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', e => {
      if (!pointer) return;
      change((e.clientX-pointer[0])*.008, (e.clientY-pointer[1])*.008);
      pointer = [e.clientX, e.clientY];
    });
    const stop = () => { pointer = undefined; };
    canvas.addEventListener('pointerup', stop); canvas.addEventListener('pointercancel', stop);
    canvas.addEventListener('lostpointercapture', stop);
    canvas.addEventListener('keydown', e => {
      const keys = {ArrowLeft: [-.15, 0], ArrowRight: [.15, 0], ArrowUp: [0, .12], ArrowDown: [0, -.12], '+': [0,0,1.2], '=': [0,0,1.2], '-': [0,0,1/1.2]};
      if (e.key === 'Home') { e.preventDefault(); reset(); }
      else if (keys[e.key]) { e.preventDefault(); change(...keys[e.key]); }
    });
    function draw() {
      if (!scene) return;
      const width = Math.max(260, Math.round(canvas.clientWidth || 650));
      const height = width < 420 ? 285 : 365, ratio = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = width*ratio; canvas.height = height*ratio; canvas.style.height = height+'px';
      const ctx = canvas.getContext('2d'); ctx.scale(ratio,ratio);
      ctx.fillStyle = '#f1f6f4'; ctx.fillRect(0,0,width,height);
      const lines = scene.lines || [], faces = scene.faces || [], points = scene.points || [];
      const vertices = [...lines.flatMap(l => l.points), ...faces.flatMap(f => f.vertices), ...points.map(p => p.position)];
      if (!vertices.length) return;
      const lo = [Infinity,Infinity,Infinity], hi = [-Infinity,-Infinity,-Infinity];
      for (const p of vertices) for (let i=0;i<3;i++) { lo[i]=Math.min(lo[i],p[i]); hi[i]=Math.max(hi[i],p[i]); }
      const centre = lo.map((v,i) => (v+hi[i])/2);
      let radius = 0;
      for (const p of vertices) radius = Math.max(radius,Math.hypot(...p.map((v,i)=>v-centre[i])));
      const scale = .38*Math.min(width,height)*zoom/(radius || 1);
      const rotate = ([x,y,z]) => {
        const a=Math.cos(yaw)*x-Math.sin(yaw)*y, b=Math.sin(yaw)*x+Math.cos(yaw)*y;
        return [a, Math.sin(pitch)*b-Math.cos(pitch)*z, Math.cos(pitch)*b+Math.sin(pitch)*z];
      };
      const project = p => { const [x,y,z]=rotate(p.map((v,i)=>v-centre[i])); return [width/2+x*scale,height*.46+y*scale,z]; };
      const primitives = [
        ...faces.map(f=>({kind:'face',coords:f.vertices.map(project),color:f.color || '#91bcb4'})),
        ...lines.flatMap(l=>l.points.slice(1).map((p,i)=>({kind:'line',coords:[project(l.points[i]),project(p)],color:l.color || '#174d59',width:l.width || 1.4}))),
        ...points.map(p=>({kind:'point',coords:[project(p.position)],color:p.color || '#ad491d',label:p.label,radius:p.radius || 4})),
      ].filter(p=>p.coords.length);
      for (const p of primitives) p.depth=p.coords.reduce((s,v)=>s+v[2],0)/p.coords.length;
      primitives.sort((a,b)=>a.depth-b.depth);
      ctx.lineJoin='round'; ctx.lineCap='round';
      for (const p of primitives) {
        ctx.fillStyle=ctx.strokeStyle=p.color; ctx.lineWidth=p.width || .5;
        if (p.kind==='point') {
          const [x,y]=p.coords[0]; ctx.beginPath(); ctx.arc(x,y,p.radius,0,Math.PI*2); ctx.fill();
          if (p.label) { ctx.fillStyle='#193a40'; ctx.font='12px system-ui'; ctx.fillText(p.label,x+7,y-7); }
        } else {
          ctx.beginPath(); p.coords.forEach(([x,y],i)=>i ? ctx.lineTo(x,y) : ctx.moveTo(x,y));
          if (p.kind==='face') { ctx.closePath(); ctx.fill(); ctx.stroke(); } else ctx.stroke();
        }
      }
      // A fixed-size triad makes orientation legible even when the user zooms in.
      const base=[48,height-42];
      [[1,0,0],[0,1,0],[0,0,1]].forEach((axis,i)=>{
        const p=rotate(axis),x=base[0]+p[0]*28,y=base[1]+p[1]*28;
        ctx.strokeStyle=ctx.fillStyle=['#ad491d','#174d59','#695082'][i]; ctx.lineWidth=2;
        ctx.beginPath();ctx.moveTo(...base);ctx.lineTo(x,y);ctx.stroke();
        ctx.font='12px system-ui';ctx.fillText(['x','y','z'][i],x+3,y-3);
      });
      ctx.fillStyle='#46605e';ctx.font='11px system-ui';ctx.textAlign='right';ctx.fillText('Orthographic view',width-12,height-12);ctx.textAlign='left';
    }
    new ResizeObserver(draw).observe(figure);
    return {update(value) {
      scene=value; caption.textContent=scene.caption || '';
      if(scene.axisLabels) caption.textContent+=' Axes: '+scene.axisLabels.map((s,i)=>['x','y','z'][i]+' = '+s).join('; ')+'.';
      canvas.setAttribute('aria-label', caption.textContent+' Arrow keys rotate; plus and minus zoom; Home resets the view.');
      draw();
    }, reset};
  };
})();
