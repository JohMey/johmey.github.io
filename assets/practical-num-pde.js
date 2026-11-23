(() => {
  'use strict';
  const studies = window.fundamentalStudies ||= {};
  const fmt = (value,digits) => value.toFixed(digits);

  function curve(name, start, end, fn, count=120) {
    return {name,points:Array.from({length:count+1},(_,i)=>{
      const x=start+(end-start)*i/count;
      return [x,fn(x)];
    })};
  }

  studies['num-pde-neumann-nullspace'] = {
    controls:[['Prescribed discrete mean',-0.5,0.5,0.025,0],['Right flux minus 2',-0.8,0.8,0.05,0],['Number of cells',4,24,2,4]],
    xLabel:'Cell-centre position',yLabel:'Diffusion potential',
    calculate([mean,delta,N]) {
      const h=1/N;
      function reconstruct(source,name) {
        // March the conservative face flux from the measured left boundary.
        let q=0;
        const centres=[0];
        for(let i=1;i<N;i++) {
          q+=h*source;
          centres.push(centres[i-1]-h*q);
        }
        const offset=mean-centres.reduce((a,b)=>a+b,0)/N;
        return {name,points:centres.map((u,i)=>[(i+0.5)*h,u+offset])};
      }
      const matches=Math.abs(delta)<1e-10;
      return {curves:[reconstruct(2,'Keep original source and left flux'),reconstruct(2+delta,'Enforce both fluxes with adjusted source')],
        result:'Global source minus measured outflow is '+fmt(-delta,3)+'. '+(matches?'The data are compatible. The two reconstructions coincide.':'The original data have no steady solution. Enforcing both fluxes changes the source from 2 to '+fmt(2+delta,2)+'.')+' Both curves are shifted to a discrete mean of '+fmt(mean,3)+'.'};
    }
  };

  function solveLinearElements(nodes,contrast) {
    const cells=nodes.length-1,free=cells-1;
    const diag=Array(free).fill(0),off=Array(Math.max(0,free-1)).fill(0),force=Array(free).fill(0);
    const lengths=[],loads=[];
    for(let i=0;i<cells;i++) {
      const h=nodes[i+1]-nodes[i],f=nodes[i]<0.5?1:contrast;
      lengths.push(h);loads.push(f);
      if(i>0) { diag[i-1]+=1/h;force[i-1]+=f*h/2; }
      if(i+1<cells) { diag[i]+=1/h;force[i]+=f*h/2; }
      if(i>0&&i+1<cells) off[i-1]=-1/h;
    }
    // Dirichlet end values vanish. Eliminate the tridiagonal interior system.
    for(let i=1;i<free;i++){
      const mult=off[i-1]/diag[i-1];
      diag[i]-=mult*off[i-1];force[i]-=mult*force[i-1];
    }
    const values=Array(cells+1).fill(0);
    for(let i=free-1;i>=0;i--) values[i+1]=(force[i]-(i+1<free?off[i]*values[i+2]:0))/diag[i];
    const slopes=lengths.map((h,i)=>(values[i+1]-values[i])/h);
    const indicators=lengths.map((h,i)=>loads[i]**2*h**3);
    for(let i=1;i<cells;i++) {
      const halfJumpTerm=0.5*Math.min(lengths[i],lengths[i-1])*(slopes[i]-slopes[i-1])**2;
      indicators[i]+=halfJumpTerm;indicators[i-1]+=halfJumpTerm;
    }
    const exactError=Math.sqrt(lengths.reduce((sum,h,i)=>sum+loads[i]**2*h**3/12,0));
    return {values,indicators,exactError};
  }

  studies['num-pde-residual-refinement'] = {
    controls:[['Refinement rounds',0,6,1,1],['Right-hand source',2,16,1,9],['Bulk marking fraction',0.3,0.9,0.05,0.5]],
    xLabel:'Position along interval',yLabel:'Solution value',
    calculate([rounds,contrast,theta]) {
      let nodes=[0,0.5,1],solution=solveLinearElements(nodes,contrast);
      for(let k=0;k<rounds;k++) {
        const order=solution.indicators.map((v,i)=>[v,i]).sort((a,b)=>b[0]-a[0]);
        const goal=theta*solution.indicators.reduce((a,b)=>a+b,0);
        const selected=new Set();
        let sum=0;
        for(const [value,i] of order){selected.add(i);sum+=value;if(sum>=goal) break;}
        const split=[nodes[0]];
        for(let i=0;i<nodes.length-1;i++){
          if(selected.has(i))split.push((nodes[i]+nodes[i+1])/2);
          split.push(nodes[i+1]);
        }
        nodes=split;solution=solveLinearElements(nodes,contrast);
      }
      const c=(3+contrast)/8, exact=x=>c*x-x*x/2-(contrast-1)*Math.max(0,x-0.5)**2/2;
      return {curves:[curve('Exact quadratic pieces',0,1,exact),{name:'Adaptive linear elements',points:nodes.map((x,i)=>[x,solution.values[i]])}],
        result:'After '+rounds+' refinement round'+(rounds===1?'':'s')+' there are '+(nodes.length-1)+' cells and '+(nodes.length-2)+' free values. Exact energy error is '+fmt(solution.exactError,4)+'. The global residual indicator is '+fmt(Math.sqrt(solution.indicators.reduce((a,b)=>a+b,0)),4)+'.'};
    }
  };

  studies['num-pde-fourier-diffusion'] = {
    controls:[['Periodic grid points',8,64,2,16],['Physical cosine mode',1,16,1,3],['Diffusivity',0.1,1,0.05,0.5]],
    xLabel:'Time',yLabel:'Mode amplitude',
    calculate([N,m,kappa]) {
      const remainder=m%N,k=Math.min(remainder,N-remainder),h=2*Math.PI/N;
      const rate=4*Math.sin(k*h/2)**2/h**2;
      const decay=(kSquared,t)=>Math.exp(-kappa*kSquared*t);
      const tmax=2/(kappa*Math.max(1,k**2));
      return {curves:[curve('Intended continuum mode',0,tmax,t=>decay(m**2,t)),curve('Fourier grid',0,tmax,t=>decay(k**2,t)),curve('Centred differences',0,tmax,t=>decay(rate,t))],
        result:'The grid represents mode '+m+' as '+k+'. At t = 0.20, amplitudes are '+fmt(decay(m**2,0.2),5)+' for the physical mode, '+fmt(decay(k**2,0.2),5)+' for the Fourier grid and '+fmt(decay(rate,0.2),5)+' for the difference stencil.'};
    }
  };
})();
