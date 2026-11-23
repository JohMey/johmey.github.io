// Small, explicit browser models of the equations derived in the linked chapters.
// Downloadable OneSimulation notebooks provide the independent solver calculation.
(() => {
  "use strict";
  const studies = window.fundamentalStudies ||= {};
  const grid = (a,b,n=180) => Array.from({length:n+1},(_,i)=>a+(b-a)*i/n);
  const curve = (name,x,f) => ({name,points:x.map(t=>[t,f(t)])});
  const fmt = x => Number(x.toPrecision(6)).toString();
  const register = (key,controls,xLabel,yLabel,calculate) => {
    studies["explore-physics-"+key] = {controls,xLabel,yLabel,calculate};
  };
  function rk4(y,t,h,rates) {
    const a=rates(y,t), b=rates(y.map((v,i)=>v+h*a[i]/2),t+h/2);
    const c=rates(y.map((v,i)=>v+h*b[i]/2),t+h/2), d=rates(y.map((v,i)=>v+h*c[i]),t+h);
    return y.map((v,i)=>v+h*(a[i]+2*b[i]+2*c[i]+d[i])/6);
  }

  register("drag",[["Terminal speed (m/s)",.5,8,.1,3],["Effective gravity (m/s²)",1,15,.01,9.81],
    ["RK4 time step (s)",.005,.05,.005,.01]],"Time after release (s)","Downward speed (m/s)",([v,g,step])=>{
    const count=Math.ceil(1.5/step), h=1.5/count, times=grid(0,1.5,count);
    const linear=[], quadratic=[], exact=[];
    let state=[0,0], error=0;
    const rates=([a,b])=>[g*(1-a/v),g*(1-b*Math.abs(b)/(v*v))];
    times.forEach((t,i)=>{
      if(i) state=rk4(state,t-h,h,rates);
      const reference=v*Math.tanh(g*t/v);
      linear.push([t,state[0]]); quadratic.push([t,state[1]]); exact.push([t,reference]);
      error=Math.max(error,Math.abs(state[1]-reference));
    });
    return {curves:[{name:"Linear drag · RK4",points:linear},{name:"Quadratic drag · RK4",points:quadratic},
      {name:"Quadratic drag · exact",points:exact}],result:"Characteristic time = "+fmt(v/g)+" s. Maximum quadratic-drag integration error = "+error.toExponential(2)+" m/s. At 0.30 s the exact speeds are "+fmt(v*(-Math.expm1(-.3*g/v)))+" and "+fmt(v*Math.tanh(.3*g/v))+" m/s. Actual step = "+fmt(h)+" s."};
  });

  register("molecule",[["Nuclear separation (bohr)",.9,8,.05,2.5]],"Position along the internuclear axis (bohr)","Electron probability density (bohr⁻³)",([r])=>{
    const s=Math.exp(-r)*(1+r+r*r/3), j=-1/r+(1+1/r)*Math.exp(-2*r), k=-(1+r)*Math.exp(-r);
    const h=-.5+j, off=-.5*s+k, even=(h+off)/(1+s)+1/r, odd=(h-off)/(1-s)+1/r;
    const x=grid(-6,6,360), orbital=(t,sign)=>{
      const a=Math.exp(-Math.abs(t+r/2)), b=Math.exp(-Math.abs(t-r/2));
      return (a+sign*b)**2/(2*Math.PI*(1+sign*s));
    };
    return {curves:[curve("Even combination |ψ+|²",x,t=>orbital(t,1)),curve("Odd combination |ψ−|²",x,t=>orbital(t,-1)),
      curve("Incoherent half-and-half density",x,t=>(Math.exp(-2*Math.abs(t+r/2))+Math.exp(-2*Math.abs(t-r/2)))/(2*Math.PI))],
      result:"Nuclei at x = ±"+fmt(r/2)+" bohr. Overlap S = "+fmt(s)+". Total even/odd energies = "+fmt(even)+" / "+fmt(odd)+" hartree. Even-state binding relative to H + proton = "+fmt((-.5-even)*27.2114)+" eV. The plotted line samples a three-dimensional density; its one-dimensional integral need not equal one."};
  });

  register("eutectic",[["Melting point of B (K)",500,1100,10,700],["Fusion enthalpy of B (kJ/mol)",5,20,.5,10],
    ["Overall mole fraction of B",.05,.95,.01,.4]],"Mole fraction of B","Temperature (K)",([tmB,hB,z])=>{
    const R=8.314462618, tmA=900, hA=14500;
    const fraction=(t,tm,h)=>Math.exp(-h/R*(1/t-1/tm));
    let low=1, high=Math.min(tmA,tmB);
    for(let i=0;i<65;i++) {
      const t=(low+high)/2;
      if(fraction(t,tmA,hA)+fraction(t,tmB,1000*hB)<1) low=t; else high=t;
    }
    const te=(low+high)/2, xe=fraction(te,tmB,1000*hB);
    const a=grid(te,tmA).map(t=>[1-fraction(t,tmA,hA),t]);
    const b=grid(te,tmB).map(t=>[fraction(t,tmB,1000*hB),t]);
    const liquid=z<=xe?z/xe:(1-z)/(1-xe), first=z<xe?"A":z>xe?"B":"neither before the eutectic";
    return {curves:[{name:"Liquid + solid A",points:a},{name:"Liquid + solid B",points:b},
      {name:"Three-phase temperature",points:[[0,te],[1,te]]},
      {name:"Selected composition",points:[[z,te],[z,Math.max(tmA,tmB)]]}],
      result:"Eutectic: "+fmt(te)+" K, xB = "+fmt(xe)+". First solid: "+first+". Liquid fraction immediately above the eutectic = "+fmt(liquid)+" (mole basis). A remains fixed at 900 K and 14.5 kJ/mol."};
  });

  // Jacobi rotations diagonalise the real symmetric 3×3 Christoffel matrix.
  // Sorting gives continuous speed branches; polarisation labels can interchange.
  function eigenvalues(a) {
    const m=a.map(row=>row.slice());
    for(let iteration=0;iteration<30;iteration++) {
      let p=0,q=1;
      for(const [i,j] of [[0,2],[1,2]]) if(Math.abs(m[i][j])>Math.abs(m[p][q])) [p,q]=[i,j];
      if(Math.abs(m[p][q])<1e-12) break;
      const angle=.5*Math.atan2(2*m[p][q],m[q][q]-m[p][p]);
      const c=Math.cos(angle),s=Math.sin(angle),app=m[p][p],aqq=m[q][q],apq=m[p][q];
      m[p][p]=c*c*app-2*c*s*apq+s*s*aqq;
      m[q][q]=s*s*app+2*c*s*apq+c*c*aqq; m[p][q]=m[q][p]=0;
      for(let k=0;k<3;k++) if(k!==p&&k!==q) {
        const a=m[k][p],b=m[k][q];
        m[k][p]=m[p][k]=c*a-s*b; m[k][q]=m[q][k]=s*a+c*b;
      }
    }
    return [m[0][0],m[1][1],m[2][2]].sort((a,b)=>a-b);
  }
  register("cubic-sound",[["Shear stiffness C44 (GPa)",30,110,1,80],["Density (kg/m³)",1800,9000,10,2330],
    ["3D branch: 0 slow, 1 middle, 2 fast",0,2,1,2]],"Direction [100] → [010] (degrees)","Phase speed (km/s)",([c44,rho,branch])=>{
    const speeds=n=>eigenvalues(n.map((ni,i)=>n.map((nj,j)=>(64+c44)*ni*nj+(i===j?c44+(102-2*c44)*ni*ni:0)))).map(v=>Math.sqrt(v*1000/rho));
    const angles=grid(0,90), names=["Slow branch","Middle branch","Fast branch"];
    const values=angles.map(a=>speeds([Math.cos(a*Math.PI/180),Math.sin(a*Math.PI/180),0]));
    const curves=names.map((name,j)=>({name,points:angles.map((a,i)=>[a,values[i][j]])}));
    const point=(theta,phi)=>{
      const n=[Math.sin(theta)*Math.cos(phi),Math.sin(theta)*Math.sin(phi),Math.cos(theta)];
      return n.map(v=>v*speeds(n)[branch]);
    };
    const latitudes=18,longitudes=36, mesh=Array.from({length:latitudes+1},(_,i)=>
      Array.from({length:longitudes},(_,j)=>point(Math.PI*i/latitudes,2*Math.PI*j/longitudes)));
    const faces=[],lines=[];
    for(let i=0;i<latitudes;i++) for(let j=0;j<longitudes;j++) faces.push({vertices:[mesh[i][j],mesh[i][(j+1)%longitudes],mesh[i+1][(j+1)%longitudes],mesh[i+1][j]],color:["#d4e6d3","#e4d8ef","#c4e0e8"][branch]});
    for(let i=3;i<latitudes;i+=3) lines.push({points:[...mesh[i],mesh[i][0]],color:"#476c75",width:.7});
    for(let j=0;j<longitudes;j+=6) lines.push({points:mesh.map(row=>row[j]),color:"#476c75",width:.7});
    const v100=speeds([1,0,0]),v110=speeds([Math.SQRT1_2,Math.SQRT1_2,0]);
    return {curves,result:"C11 = 166 GPa, C12 = 64 GPa. [100] speeds: "+v100.map(fmt).join(", ")+" km/s; [110]: "+v110.map(fmt).join(", ")+" km/s. C44 = 51 GPa makes the material isotropic. The 3D view shows the "+names[branch].toLowerCase()+".",
      scene:{faces,lines,axisLabels:["vx (km/s)","vy (km/s)","vz (km/s)"],
        caption:"Three-dimensional phase-speed surface: in each propagation direction n, the radius is the selected eigenvalue's speed v(n). Coordinates are v(n)n in km/s. This is a material property surface, not a deformed crystal or a group-velocity wavefront."}};
  });

  studies["explore-physics-cubic-sound"].localModel = {
    file: "complete-physics-cubic-surface.py",
    url: "/downloads/fundamentals/examples/complete-physics-cubic-surface.py",
    args: ([c44,density,branch]) => `--c44 ${c44} --density ${density} --branch ${branch}`
  };

  register("harmonic",[["Phase mismatch Δk·Lnl",0,8,.1,2],["Crystal length / Lnl",1,6,.1,3],
    ["RK4 step / Lnl",.005,.05,.005,.01]],"Distance / nonlinear coupling length","Power fraction",([mismatch,length,step])=>{
    const count=Math.ceil(length/step),h=length/count;
    function rates([ar,ai,br,bi],z) {
      const c=Math.cos(mismatch*z),s=Math.sin(mismatch*z);
      const ur=ar*br+ai*bi,ui=ar*bi-ai*br,vr=ar*ar-ai*ai,vi=2*ar*ai;
      return [-ur*c-ui*s,ur*s-ui*c,vr*c-vi*s,vr*s+vi*c];
    }
    let state=[1,0,0,0],error=0;
    const pump=[],harmonic=[],exact=[];
    for(let i=0;i<=count;i++) {
      const z=i*h;
      if(i) state=rk4(state,z-h,h,rates);
      const p=state[0]**2+state[1]**2,q=state[2]**2+state[3]**2;
      error=Math.max(error,Math.abs(p+q-1));
      pump.push([z,p]); harmonic.push([z,q]); exact.push([z,Math.tanh(z)**2]);
    }
    return {curves:[{name:"Fundamental power · RK4",points:pump},{name:"Second harmonic · RK4",points:harmonic},
      {name:"Exact second harmonic at Δk = 0",points:exact}],
      result:"Exit harmonic fraction = "+fmt(harmonic.at(-1)[1])+". Largest |Pω + P2ω − 1| = "+error.toExponential(2)+". Actual step = "+fmt(h)+". Normalised, lossless plane-wave model with pump depletion."};
  });

  register("orientation",[["Dipole energy / thermal energy pE/(kBT)",0,12,.1,3],["Trapezoidal integration intervals",10,200,10,40]],
    "Orientation cosine u = cos θ","Probability density per unit u",([x,n])=>{
      const langevin=x<.01?x/3-x**3/45+2*x**5/945:1/Math.tanh(x)-1/x;
      const norm=x===0?2:-Math.expm1(-2*x)/x;
      const density=u=>Math.exp(x*(u-1))/norm;
      const u=grid(-1,1,n), continuous=grid(-1,1,240);
      let denominator=0,numerator=0;
      u.forEach((v,i)=>{const weight=i===0||i===n?.5:1;denominator+=weight*density(v);numerator+=weight*v*density(v);});
      const mean=numerator/denominator;
      return {curves:[curve("Boltzmann orientation density",continuous,density),curve("Integration samples",u,density),
        curve("Zero-field isotropic density",continuous,()=>.5)],
        result:"Exact mean cos θ = "+fmt(langevin)+"; trapezoidal mean = "+fmt(mean)+"; weak-field estimate = "+fmt(x/3)+". Absolute quadrature error = "+Math.abs(mean-langevin).toExponential(2)+". Density normalisation from this grid = "+fmt(denominator*2/n)+". The downloadable notebook independently uses Gaussian quadrature through OneSimulation."};
    });
})();
