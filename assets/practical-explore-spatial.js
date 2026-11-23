(() => {
  const studies=window.fundamentalStudies ||= {};
  studies['explore-spatial-poisson-cube']={
    controls:[['Cells per direction',8,24,4,16],['Source amplitude (V)',.2,2,.1,1],['Slice position (% of depth)',0,100,5,50]],
    xLabel:'x at y = 0.5 m (m)',yLabel:'Potential on selected slice (V)',
    localModel:{file:'poisson-cube-3d.py',url:'/downloads/fundamentals/examples/poisson-cube-3d.py',args:([n,a,z])=>`--cells ${n} --amplitude ${a} --slice ${z}`},
    calculate([n,a,slice]){
      const k=Math.round(n*slice/100),z=k/n;
      const amplification=3*Math.PI**2/(12*n*n*Math.sin(Math.PI/(2*n))**2);
      const exact=(x,y,z)=>a*Math.sin(Math.PI*x)*Math.sin(Math.PI*y)*Math.sin(Math.PI*z);
      const nodal=Array.from({length:n+1},(_,i)=>[i/n,amplification*exact(i/n,.5,z)]);
      const reference=Array.from({length:161},(_,i)=>[i/160,exact(i/160,.5,z)]);
      const color=value=>{const t=Math.max(0,Math.min(1,value/2.03));return `rgb(${Math.round(234-211*t)},${Math.round(243-166*t)},${Math.round(231-142*t)})`;};
      const faces=[],lines=[];
      for(let i=0;i<n;i++)for(let j=0;j<n;j++) {
        const v=[[i/n,j/n,z],[(i+1)/n,j/n,z],[(i+1)/n,(j+1)/n,z],[i/n,(j+1)/n,z]];
        const mean=v.reduce((sum,p)=>sum+amplification*exact(...p),0)/4;
        faces.push({vertices:v,color:color(mean)});
      }
      for(let dim=0;dim<3;dim++)for(let b=0;b<=1;b++)for(let c=0;c<=1;c++) {
        const p=[b,c];p.splice(dim,0,0);const q=p.slice();q[dim]=1;
        lines.push({points:[p,q],color:'#6c8b87',width:1.6});
      }
      const error=a*(amplification-1);
      return {curves:[{name:'Seven-point FDM · nodal slice',points:nodal},{name:'Exact 3D solution · same slice',points:reference}],
        result:`${n}³ cells; selected grid plane z = ${z.toFixed(3)} m. Maximum nodal error in the whole cube ${error.toExponential(3)} V (${(100*(amplification-1)).toFixed(3)}%). All six boundary potentials are zero.`,
        scene:{faces,lines,points:[{position:[0,0,0],label:'0'},{position:[1,1,1],label:'(1,1,1) m'}],
          axisLabels:['position (m)','position (m)','position (m)'],
          caption:`A slice through the three-dimensional potential, at z = ${z.toFixed(3)} m. Colour is potential: pale = 0 V, dark green = 2.03 V; slice values range from 0 to ${(a*amplification*Math.sin(Math.PI*z)).toFixed(3)} V. Grid-face colours average their four nodal values.`}};
    }
  };
})();
