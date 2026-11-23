// Small, explicit numerical experiments. The linked notebooks run the corresponding
// OneSimulation solvers; these browser previews recompute the equations locally.
(() => {
  const studies = window.fundamentalStudies ||= {};
  const grid = (a,b,n) => Array.from({length:n+1}, (_,i) => a+(b-a)*i/n);
  const curve = (name,x,f) => ({name,points:x.map((t,i)=>[t,f(t,i)])});
  const fmt = x => Number(x.toPrecision(5)).toString();
  const logError = x => Math.log10(Math.max(1e-16,Math.abs(x)));
  const add = (key,controls,xLabel,yLabel,calculate) => {
    studies['explore-numerics-'+key] = {controls,xLabel,yLabel,calculate};
  };

  add('singular-value', [['Selected log₁₀ δ',-12,-1,.25,-8],['Arithmetic bits',32,64,32,64]],
    'log₁₀ sensor separation δ','Computed / exact small singular value',([power,bits])=>{
      const round = bits===32 ? Math.fround : x=>x;
      const ratio = p => {
        const d=round(10**p), square=round(d*d);
        const eigenvalue=round(round(1+square)-round(1-square));
        return round(Math.sqrt(Math.max(0,eigenvalue)))/(Math.SQRT2*10**p);
      };
      const xs=grid(-12,-1,220), selected=ratio(power);
      return {curves:[curve('Stable formula from orthogonal rows',xs,()=>1),
        curve('Rounded Gram-matrix difference',xs,ratio),
        {name:'Selected separation',points:[[power,0],[power,Math.max(1,selected)]]}],
        result:'At δ = '+fmt(10**power)+', exact σmin = '+fmt(Math.SQRT2*10**power)+
          ', condition number = '+fmt(10**(-power))+'. The '+bits+'-bit Gram calculation retains '+
          fmt(selected)+' of the small singular value. The reference curve uses the analytic row-orthogonality formula, not an implemented SVD.'};
    });

  add('newton-certificate', [['Initial x₁',.3,3,.05,1.5],['Initial x₂',.3,2,.05,.9],
    ['log₁₀ first-equation scale',-6,6,1,4]],'Newton iteration','Distance from initial point (maximum norm)',([a,b,power])=>{
      const beta=Math.max(Math.abs((a*a-2)/(2*a)),Math.abs((b*b-1)/(2*b)));
      const lipschitz=Math.max(1/a,1/b), h=beta*lipschitz;
      const radius=h<=.5 ? 2*beta/(1+Math.sqrt(1-2*h)) : null;
      function run(scale) {
        let x=a,y=b; const points=[[0,0]];
        for(let k=1;k<=7;k++) {
          x-=scale*(x*x-2)/(scale*2*x); y-=(y*y-1)/(2*y);
          points.push([k,Math.max(Math.abs(x-a),Math.abs(y-b))]);
        }
        return points;
      }
      const original=run(1), scaled=run(10**power);
      const curves=[{name:'Original equations',points:original},{name:'Rescaled first equation',points:scaled}];
      if(radius!==null) curves.push({name:'Certified radius',points:[[0,radius],[7,radius]]});
      return {curves,result:'β = '+fmt(beta)+', L = '+fmt(lipschitz)+', βL = '+fmt(h)+'. '+
        (radius===null ? 'The sufficient condition βL ≤ 1/2 fails; this certificate makes no claim about convergence.' :
          'The certified radius is '+fmt(radius)+'. All displayed iterates remain in that ball.')+
        ' Rescaling a whole equation and its Jacobian leaves the Newton step unchanged up to rounding.'};
    });

  function gaussLegendre(n) {
    const nodes=[],weights=[];
    for(let i=0;i<n;i++) {
      let x=Math.cos(Math.PI*(i+.75)/(n+.5)), derivative=0;
      for(let k=0;k<24;k++) {
        let p0=1,p1=x;
        for(let j=2;j<=n;j++) {const next=((2*j-1)*x*p1-(j-1)*p0)/j;p0=p1;p1=next;}
        derivative=n*(x*p1-p0)/(x*x-1);
        const step=p1/derivative; x-=step;
        if(Math.abs(step)<2e-15) break;
      }
      // Re-evaluate the derivative at the final root for its quadrature weight.
      let p0=1,p1=x;
      for(let j=2;j<=n;j++) {const next=((2*j-1)*x*p1-(j-1)*p0)/j;p0=p1;p1=next;}
      derivative=n*(x*p1-p0)/(x*x-1);
      nodes.push((x+1)/2); weights.push(1/((1-x*x)*derivative*derivative));
    }
    return {nodes,weights};
  }
  add('triangle-cubature', [['Corner exponent p',0,1.75,.05,.5],['Maximum Gauss order',2,12,1,6]],
    'Number of function values n²','log₁₀ absolute error (floor −16)',([p,maxOrder])=>{
      const exact=1/(2-p), cartesian=[],radial=[]; let lastA=0,lastB=0;
      for(let n=1;n<=maxOrder;n++) {
        const {nodes,weights}=gaussLegendre(n); let a=0,b=0;
        for(let i=0;i<n;i++) for(let j=0;j<n;j++) {
          const u=nodes[i],v=nodes[j],weight=weights[i]*weights[j];
          a+=weight*(1-u)*(u+(1-u)*v)**(-p);
          b+=weight*2*u**(3-2*p);
        }
        lastA=Math.abs(a-exact);lastB=Math.abs(b-exact);
        cartesian.push([n*n,logError(lastA)]);radial.push([n*n,logError(lastB)]);
      }
      return {curves:[{name:'Cartesian triangle map',points:cartesian},{name:'Squared radial map',points:radial}],
        result:'Exact integral 1/(2 − p) = '+fmt(exact)+'. With '+maxOrder**2+' evaluations, errors are '+fmt(lastA)+
          ' (Cartesian) and '+fmt(lastB)+' (radial). For p = 1/2 the radial integrand is 2u², so order 2 already integrates it to rounding accuracy.'};
    });

  add('parasitic-root', [['Parasitic root r',-1,2,.1,2],['Steps to t = 1',6,60,2,26],
    ['log₁₀ starting perturbation',-14,-3,1,-7]],'Time t','log₁₀ absolute error or change (floor −16)',([r,n,power])=>{
      const h=1/n,starter=1-h+h*h/2-h**3/6+h**4/24;
      let previous=1,current=starter,deltaPrevious=0,deltaCurrent=10**power;
      const clean=[[0,-16],[h,logError(current-Math.exp(-h))]],
        perturbed=[[0,-16],[h,logError(current+deltaCurrent-Math.exp(-h))]],
        difference=[[0,-16],[h,logError(deltaCurrent)]];
      for(let k=1;k<n;k++) {
        const coefficient=1+r-h*(1-r);
        const next=coefficient*current-r*previous,deltaNext=coefficient*deltaCurrent-r*deltaPrevious;
        previous=current;current=next;deltaPrevious=deltaCurrent;deltaCurrent=deltaNext;
        const t=(k+1)*h,reference=Math.exp(-t);
        clean.push([t,logError(current-reference)]);
        perturbed.push([t,logError(current-reference+deltaCurrent)]);
        difference.push([t,logError(deltaCurrent)]);
      }
      const stable=r>=-1&&r<1;
      return {curves:[{name:'Unperturbed discretisation error',points:clean},
        {name:'Error with perturbed start',points:perturbed},{name:'Propagated starting change',points:difference}],
        result:'The characteristic roots at h = 0 are 1 and '+fmt(r)+'. '+
          (stable?'The root condition holds.':r===1?'A repeated unit root violates the root condition.':'The parasitic root lies outside the unit circle.')+
          ' A starting change of '+fmt(10**power)+' produces '+fmt(Math.abs(deltaCurrent))+' change at t = 1. The error-change recurrence is evaluated separately to avoid subtracting two nearly equal trajectories.'};
    });

  function instrument(receivers,sources,width) {
    return receivers.map(s=>{
      const row=sources.map(t=>{const d=((s-t+.5)%1+1)%1-.5;return Math.exp(-.5*(d/width)**2);});
      const sum=row.reduce((a,b)=>a+b,0);return row.map(x=>x/sum);
    });
  }
  function choleskySolve(a,b) {
    const n=b.length,l=Array.from({length:n},()=>Array(n).fill(0));
    for(let i=0;i<n;i++) for(let j=0;j<=i;j++) {
      let sum=a[i][j];for(let k=0;k<j;k++)sum-=l[i][k]*l[j][k];
      l[i][j]=i===j?Math.sqrt(sum):sum/l[j][j];
    }
    const y=[],x=Array(n);
    for(let i=0;i<n;i++){let sum=b[i];for(let j=0;j<i;j++)sum-=l[i][j]*y[j];y[i]=sum/l[i][i];}
    for(let i=n-1;i>=0;i--){let sum=y[i];for(let j=i+1;j<n;j++)sum-=l[j][i]*x[j];x[i]=sum/l[i][i];}
    return x;
  }
  const noisePattern=(()=>{
    let seed=17;
    const uniform=()=>{seed=(Math.imul(1664525,seed)+1013904223)>>>0;return (seed+.5)/4294967296;};
    return Array.from({length:32},()=>Math.sqrt(-2*Math.log(uniform()))*Math.cos(2*Math.PI*uniform()));
  })();
  add('inverse-blur', [['True blur width',.02,.12,.005,.065],['Assumed blur width',.02,.12,.005,.045],
    ['Noise standard deviation',0,.02,.001,.002],['log₁₀ regularisation μ',-5,0,.25,-2]],
    'Position on periodic interval','Signal / measurement',([trueWidth,assumedWidth,noise,power])=>{
      const n=32,coarse=Array.from({length:n},(_,i)=>(i+.5)/n),fine=Array.from({length:1024},(_,i)=>(i+.5)/1024);
      const signal=x=>Math.cos(2*Math.PI*x)+.4*Math.sin(10*Math.PI*x),truth=coarse.map(signal),fineTruth=fine.map(signal);
      const a=instrument(coarse,coarse,assumedWidth),forward=instrument(coarse,fine,trueWidth);
      const b=forward.map((row,i)=>row.reduce((s,v,j)=>s+v*fineTruth[j],0)+noise*noisePattern[i]);
      const mu=10**power;
      const normal=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>
        a.reduce((sum,row)=>sum+row[i]*row[j],0)+(i===j?mu*mu:0)));
      const rhs=Array.from({length:n},(_,i)=>a.reduce((sum,row,j)=>sum+row[i]*b[j],0));
      const reconstruction=choleskySolve(normal,rhs);
      const rms=Math.sqrt(reconstruction.reduce((s,v,i)=>s+(v-truth[i])**2,0)/n);
      const residual=Math.sqrt(a.reduce((s,row,i)=>s+(row.reduce((t,v,j)=>t+v*reconstruction[j],0)-b[i])**2,0)/n);
      return {curves:[curve('True signal',grid(0,1,200),signal),curve('Noisy instrument data',coarse,(_,i)=>b[i]),
        curve('Tikhonov reconstruction',coarse,(_,i)=>reconstruction[i])],
        result:'Reconstruction RMS error = '+fmt(rms)+'; data-fit RMS residual = '+fmt(residual)+
          '; μ = '+fmt(mu)+'. Data use 1024 source cells; the inverse model uses 32. The fixed browser noise realisation is reproducible but differs from the notebook seed implementation.'};
    });

  add('adi-heat', [['Interior grid points per direction',5,31,2,9],['Final time',0,.3,.01,.1],
    ['Time steps',1,64,1,8],['Vertical mode number q',1,4,1,1]],
    'Horizontal position x','Temperature / initial amplitude',([m,time,n,q])=>{
      const spacing=1/(m+1),h=time/n;
      const lx=-4*Math.sin(Math.PI*spacing/2)**2/spacing**2;
      const ly=-4*Math.sin(q*Math.PI*spacing/2)**2/spacing**2;
      const factor=l=>(1+h*l/2)/(1-h*l/2);
      const amplification=(factor(lx)*factor(ly))**n;
      const semiDiscrete=Math.exp((lx+ly)*time),continuum=Math.exp(-(Math.PI**2)*(1+q*q)*time);
      const xs=grid(0,1,m+1),dense=grid(0,1,120);
      const field=(x,y)=>amplification*Math.sin(Math.PI*x)*Math.sin(q*Math.PI*y);
      const faces=[],lines=[];
      for(let i=0;i<=m;i++) for(let j=0;j<=m;j++) {
        const corners=[[xs[i],xs[j]],[xs[i+1],xs[j]],[xs[i+1],xs[j+1]],[xs[i],xs[j+1]]];
        const value=corners.reduce((s,[x,y])=>s+field(x,y),0)/4;
        faces.push({vertices:corners.map(([x,y])=>[x,y,field(x,y)]),color:`hsl(${200-155*(value+1)/2} 48% 63%)`});
      }
      for(let i=0;i<xs.length;i++) {
        lines.push({points:xs.map(y=>[xs[i],y,field(xs[i],y)]),color:'#174d59',width:.7});
        lines.push({points:xs.map(x=>[x,xs[i],field(x,xs[i])]),color:'#174d59',width:.7});
      }
      // A frame at the initial amplitude preserves visual height when heat decays.
      lines.push({points:[[0,0,-1],[0,0,1]],color:'#a3ada9',width:1});
      return {curves:[curve('ADI grid mode, antinode slice',xs,x=>amplification*Math.sin(Math.PI*x)),
        curve('Exact time evolution on this grid',dense,x=>semiDiscrete*Math.sin(Math.PI*x)),
        curve('Continuum heat equation',dense,x=>continuum*Math.sin(Math.PI*x))],
        result:'ADI amplitude = '+fmt(amplification)+'. Temporal amplitude error = '+fmt(Math.abs(amplification-semiDiscrete))+
          '; spatial amplitude error = '+fmt(Math.abs(semiDiscrete-continuum))+'. The slice is at y = 1/(2q) = '+fmt(1/(2*q))+
          '. Diffusivity and side length are one. '+(amplification<0?'The coarse step reverses the mode sign: stability has not ensured correct damping.':''),
        scene:{faces,lines,points:[],axisLabels:['x / L','y / L','u / u₀'],
          caption:'Three-dimensional graph of a two-dimensional heat equation: height and colour show temperature, not a third spatial coordinate. The mesh is the ADI numerical solution. Rotate it to see the q vertical lobes; the fixed −1 to +1 height reference makes decay visible.'}};
    });
})();
