// These small problems expose an exact check alongside the numerical approximation.
(() => {
  const studies = window.fundamentalStudies ||= {};
  const grid = (a,b,n) => Array.from({length:n+1},(_,i)=>a+(b-a)*i/n);
  const curve = (name,x,f) => ({name,points:x.map((t,i)=>[t,f(t,i)])});
  const label = x => Number(x.toPrecision(5)).toString();
  const logError = x => Math.log10(Math.max(1e-15,Math.abs(x)));
  function add(key,controls,xLabel,yLabel,calculate) {
    studies['num-'+key]={controls,xLabel,yLabel,calculate};
  }

  add('positive-pivots',[['Coupling c',-.99,.99,.01,.6]],'First displacement s','Quadratic form xᵀAx',([c])=>{
    const xs=grid(-2,2,100);
    return {curves:[-1,0,1].map(z=>curve('Second displacement '+z,xs,s=>s*s+2*c*s*z+z*z)),
      result:'Second pivot '+label(1-c*c)+'. For load (1, −1), first displacement '+label(1/(1-c))+'.'};
  });
  add('energy-preconditioning',[['Direct stiffness ratio',1,100,1,25],['Dimensionless coupling',0,.99,.01,.3]],
    'Iteration','Worst-case error / initial error',([a,c])=>{
      const spread=Math.hypot(a-1,2*c*Math.sqrt(a)), unscaled=spread/(a+1);
      const k=grid(0,35,35);
      return {curves:[curve('Original coordinates',k,i=>unscaled**i),curve('Diagonally scaled',k,i=>c**i)],
        result:'Optimal fixed-step contraction '+label(unscaled)+' before scaling and '+label(c)+' afterwards.'};
    });
  add('rayleigh-stationarity',[['Separation of squared frequencies (s⁻²)',.1,10,.1,5],['Trial angle (degrees)',0,85,1,10]],
    'Angle from lowest exact mode (degrees)','Squared frequency / residual (s⁻²)',([gap,a])=>{
      const x=grid(0,90,90), err=t=>gap*Math.sin(t*Math.PI/180)**2,
        res=t=>gap*Math.abs(Math.sin(t*Math.PI/180)*Math.cos(t*Math.PI/180));
      return {curves:[curve('Quotient error',x,err),curve('Residual norm',x,res),{name:'Selected angle',points:[[a,0],[a,err(a)]]}],
        result:'Selected angle gives quotient error '+label(err(a))+' s⁻² and residual '+label(res(a))+' s⁻².'};
    });

  function interpolant(xs,ys) {
    const weights=xs.map((v,i)=>1/xs.reduce((p,w,j)=>i===j?p:p*(v-w),1));
    return t=>{
      const hit=xs.indexOf(t);
      if(hit!==-1) return ys[hit];
      let top=0,bottom=0;
      for(let j=0;j<xs.length;j++) {const v=weights[j]/(t-xs[j]);top+=v*ys[j];bottom+=v;}
      return top/bottom;
    };
  }
  add('barycentric-weights',[['Polynomial degree',3,22,1,10]],'Position x','Interpolated value',([n])=>{
    const f=t=>1/(1+25*t*t), x=grid(-1,1,500), even=grid(-1,1,n), clustered=grid(0,Math.PI,n).map(Math.cos);
    const p=interpolant(even,even.map(f)), q=interpolant(clustered,clustered.map(f));
    const maxError=g=>Math.max(...x.map(t=>Math.abs(g(t)-f(t))));
    return {curves:[curve('Target',x,f),curve('Uniform nodes',x,p),curve('Cosine nodes',x,q)],
      result:'Maximum observed error on 501 test points: uniform '+label(maxError(p))+', cosine '+label(maxError(q))+'.'};
  });
  add('periodic-aliasing',[['Sensors N',3,18,1,8],['Physical harmonic k',1,24,1,8]],'Sensor-ring offset (degrees)','Mean temperature (°C)',([n,k])=>{
    const mean=a=>Array.from({length:n},(_,j)=>20+3*Math.cos(k*(2*Math.PI*j/n+a*Math.PI/180))).reduce((s,t)=>s+t,0)/n;
    const x=grid(0,360,360);
    return {curves:[curve('Measured mean',x,mean),curve('True mean',x,()=>20)],
      result:'At zero offset the sensors read '+label(mean(0))+' °C. The largest possible mean error for this grid is '+(k%n===0?'3':'0')+' °C in exact arithmetic.'};
  });
  add('richardson-tableau',[['Power p',.25,4,.25,.5]],'Refinement k (2ᵏ panels)','log₁₀ absolute error, floor −15',([p])=>{
    const exact=1/(p+1),levels=grid(0,9,9), t=levels.map(k=>{
      const n=2**k;let sum=.5;for(let j=1;j<n;j++)sum+=(j/n)**p;return sum/n;
    });
    const r=levels.slice(1).map((k)=>[k,logError((4*t[k]-t[k-1])/3-exact)]);
    return {curves:[curve('Trapezoidal',levels,(_,k)=>logError(t[k]-exact)),{name:'First extrapolation',points:r}],
      result:'Exact integral '+label(exact)+'. At 512 panels, trapezoidal error '+label(Math.abs(t[9]-exact))+' and extrapolated error '+label(Math.abs((4*t[9]-t[8])/3-exact))+'.'};
  });

  add('multiple-roots',[['Root multiplicity',1,8,1,3],['Initial distance to root',.05,1,.05,.3]],
    'Iteration','log₁₀ location error, floor −15',([m,start])=>{
      function iterate(factor){
        let e=start;const points=[[0,logError(e)]];
        for(let k=1;k<=12;k++){e-=factor*e*(1+e)/(m*(1+e)+e);points.push([k,logError(e)]);}
        return points;
      }
      return {curves:[{name:'Newton',points:iterate(1)},{name:'Multiplicity corrected',points:iterate(m)}],
        result:'Asymptotic factor for ordinary Newton: '+label(1-1/m)+'. For the corrected step the linear term vanishes.'};
    });
  add('aitken-acceleration',[['Local slope q',.1,.95,.05,.75],['Initial error',.05,.45,.05,.3]],
    'Map evaluations','log₁₀ location error, floor −15',([q,start])=>{
      // Work in error coordinates to isolate the iteration from subtraction of the known fixed point.
      const map=e=>q*e+.05*e*e;
      let p=start,a=start;const plain=[[0,logError(p)]],fast=[[0,logError(a)]];
      for(let k=1;k<=20;k++){
        p=map(p);plain.push([k,logError(p)]);
        if(k%2===0){const t=map(a),u=map(t),diff=u-2*t+a;if(diff!==0)a-=(t-a)**2/diff;fast.push([k,logError(a)]);}
      }
      return {curves:[{name:'Plain substitution',points:plain},{name:'Steffensen',points:fast}],
        result:'After twenty map evaluations, plain error '+label(Math.abs(p))+' and accelerated error '+label(Math.abs(a))+'.'};
    });
  add('trust-radius',[['Starting value',.1,1.3,.05,.2],['Trust radius',.1,5,.1,1]],
    'Proposed forward step s','Merit (residual squared / 2)',([x,delta])=>{
      const f=x*x-2, step=Math.min(-f/(2*x),delta), merit=s=>((x+s)**2-2)**2/2, model=s=>(f+2*x*s)**2/2;
      const limit=Math.max(1.2*step,.5),xs=grid(0,limit,120), ratio=(merit(0)-merit(step))/(model(0)-model(step));
      return {curves:[curve('True merit',xs,merit),curve('Linearised residual model',xs,model),{name:'Selected step',points:[[step,model(step)],[step,merit(step)]]}],
        result:'Proposed step '+label(step)+'. Actual / predicted decrease '+label(ratio)+'.'};
    });

  add('discrepancy-and-gcv',[['Error in weakest observation',-.04,.08,.002,.02]],'log₁₀ regularisation λ','Euclidean norm',([noise])=>{
    const xs=grid(-4,0,160),sigma=[1,.2,.02],d=[1,.2,.02+noise];let best=Infinity,bestLambda,bestResidual;
    const values=xs.map(t=>{
      const a=10**(2*t), h=sigma.map(s=>s*s/(s*s+a)), residual=Math.hypot(...d.map((v,i)=>(1-h[i])*v)),
        xerror=Math.hypot(...d.map((v,i)=>v*h[i]/sigma[i]-1)),gcv=3*residual**2/(3-h.reduce((sum,v)=>sum+v,0))**2;
      if(gcv<best){best=gcv;bestLambda=Math.sqrt(a);bestResidual=residual;}return [xerror,residual];
    });
    return {curves:[curve('Parameter error (truth known here)',xs,(_,i)=>values[i][0]),curve('Data residual',xs,(_,i)=>values[i][1])],
      result:'Grid minimum GCV '+label(best)+' at λ = '+label(bestLambda)+', giving residual '+label(bestResidual)+'. Actual data noise norm '+label(Math.abs(noise))+'.'};
  });
  add('sequential-assimilation',[['Assumed process variance Q (°C²)',0,2,.02,.2],['Assumed sensor variance R (°C²)',.1,5,.1,1]],
    'Observation number','Temperature (°C)',([q,r])=>{
      let mean=20,p=4,gain=0;const xs=grid(1,48,47), truth=xs.map(k=>k<20?20:23),
        data=xs.map((k,i)=>truth[i]+.8*Math.sin(2.1*k)+.42*Math.cos(.77*k)), est=data.map(z=>{
          p+=q;gain=p/(p+r);mean+=gain*(z-mean);p*=1-gain;return mean;});
      return {curves:[curve('True temperature',xs,(_,i)=>truth[i]),curve('Sensor trace',xs,(_,i)=>data[i]),curve('Posterior mean',xs,(_,i)=>est[i])],
        result:'Last posterior mean '+label(mean)+' °C, standard deviation '+label(Math.sqrt(p))+' °C. Gain '+label(gain)+'.'};
    });
  add('symmetric-posteriors',[['Squared-amplitude datum d',-.2,2,.05,1],['Signed sensor reading b',-2,2,.05,.5]],
    'Candidate parameter x','Posterior density',([d,b])=>{
      const xs=grid(-3,3,480), dx=xs[1]-xs[0], w=xs.map(x=>Math.exp(-.5*((d-x*x)/.2)**2-.5*(x/2)**2)),
        both=w.map((v,i)=>v*Math.exp(-.5*((b-xs[i])/.5)**2));
      const integral=v=>v.slice(1).reduce((s,y,i)=>s+dx*(y+v[i])/2,0);
      const z1=integral(w),z2=integral(both),positive=integral(both.slice(240))/z2;
      return {curves:[curve('Squared amplitude only',xs,(_,i)=>w[i]/z1),curve('With signed sensor',xs,(_,i)=>both[i]/z2)],
        result:'Probability of positive x on [−3,3]: 50% before the signed measurement, '+label(positive*100)+'% afterwards.'};
    });
  add('fitting-a-rotation',[['Marker half-baseline (mm)',.5,8,.1,2],['Vertical error of right marker (mm)',-.8,.8,.02,.2]],'Trial rotation (degrees)','Sum squared errors (mm²)',([a,dy])=>{
    const angle=20*Math.PI/180, x=2*a*Math.cos(angle),y=2*a*Math.sin(angle)+dy, fit=Math.atan2(y,x)*180/Math.PI;
    const E=d=>((2*a*Math.cos(d*Math.PI/180)-x)**2+(2*a*Math.sin(d*Math.PI/180)-y)**2)/2;
    return {curves:[curve('Squared fit residual',grid(-35,75,220),E),{name:'Optimal angle',points:[[fit,E(fit)]]}],
      result:'Recovered angle '+label(fit)+'°, error '+label(fit-20)+'°. Sum squared residuals '+label(E(fit))+' mm².'};
  });
  add('network-resistance',[['A–B resistance (Ω)',.5,12,.1,2],['B–C resistance (Ω)',.5,12,.1,4]],'Direct A–C resistance (Ω)','Resistance (Ω) or test voltage (V)',([r1,r2])=>{
    const x=grid(.1,16,150), total=r=>r*(r1+r2)/(r+r1+r2), vb=r=>total(r)*r2/(r1+r2);
    return {curves:[curve('Effective resistance (Ω)',x,total),curve('Voltage at B for 1 A (V)',x,vb)],
      result:'With direct edge 6 Ω, effective resistance '+label(total(6))+' Ω and interior voltage '+label(vb(6))+' V.'};
  });
  add('clenshaw-recurrence',[['Order of uncertain coefficient',2,18,1,10],['Coefficient error',-.01,.01,.0005,.001]],'Calibration coordinate x','Fitted response',([n,delta])=>{
    const x=grid(-1,1.25,225),base=t=>1+.4*t;
    function estimate(t){let b1=0,b2=0;for(let k=n;k>=1;k--){const b=(k===n?delta:k===1?.4:0)+2*t*b1-b2;b2=b1;b1=b;}return 1+t*b1-b2;}
    return {curves:[curve('Reference calibration',x,base),curve('Perturbed fit',x,estimate)],
      result:'Difference at x = 0.8: '+label(estimate(.8)-base(.8))+'. At x = 1.2: '+label(estimate(1.2)-base(1.2))+'.'};
  });
  add('importance-sampling',[['Integrand exponent b',-.75,.5,.05,-.5],['Proposal exponent a',-.8,.5,.05,-.25]],'Draws','Running integral estimate',([b,a])=>{
    const xs=grid(1,320,319),exact=1/(1+b);let seed=22517,sa=0,su=0;
    const wa=[],wu=[];
    for(const n of xs){seed=(Math.imul(seed,1664525)+1013904223)>>>0;const u=(seed+.5)/4294967296;
      sa+=u**((b-a)/(1+a))/(1+a);su+=u**b;wa.push(sa/n);wu.push(su/n);}
    const d=2*b-a+1,variance=d>1e-12?Math.max(0,1/((1+a)*d)-exact**2):Infinity;
    return {curves:[curve('Importance proposal',xs,(_,i)=>wa[i]),curve('Uniform proposal',xs,(_,i)=>wu[i]),curve('Exact integral',xs,()=>exact)],
      result:'Exact integral '+label(exact)+'. Theoretical single-weight variance of the selected proposal: '+(isFinite(variance)?label(variance):'infinite')+'.'};
  });
  add('logarithmic-unknowns',[['Exact root (positive units)',.1,20,.1,1],['Initial ratio to true root',.1,6,.05,3]],'Starting ratio x / root','Next candidate / root',([r,u])=>{
    const q=z=>z*(1-Math.log(z)),xs=grid(.05,6,180);
    return {curves:[curve('Additive candidate',xs,q),curve('Log-coordinate candidate',xs,()=>1),{name:'Selected start',points:[[u,q(u)],[u,1]]}],
      result:'Starting from '+label(r*u)+', additive Newton proposes '+label(r*q(u))+' while the log-coordinate step gives '+label(r)+'.'};
  });
  add('fold-conditioning',[['Starting displacement',-.8,0,.01,-.1],['Tangent step length',.02,.7,.01,.2]],'Displacement x','Control p',([x0,ds])=>{
    const k=-2*x0,norm=Math.hypot(1,k),xp=x0+ds/norm,pp=-x0*x0+ds*k/norm,constant=xp+k*pp;
    const disc=1-4*k*constant,root=disc>=0?2*constant/(1+Math.sqrt(disc)):null;
    const curves=[curve('Equilibria',grid(-1,1,180),x=>-x*x),{name:'Tangent predictor',points:[[x0,-x0*x0],[xp,pp]]}];
    if(root!==null)curves.push({name:'Correction',points:[[xp,pp],[root,-root*root]]});
    return {curves,result:'Predictor ('+label(xp)+', '+label(pp)+'). '+(root===null?'The correction line misses the parabola; reduce the step.':'Nearby corrected point ('+label(root)+', '+label(-root*root)+').')};
  });
  add('resolution-and-discrepancy',[['Omitted-effect norm',.05,2,.05,.5],['Direction (degrees)',0,180,1,30]],'Discrepancy direction (degrees)','Magnitude in measurement units',([b,a])=>{
    const xs=grid(0,180,180),rad=t=>t*Math.PI/180, bias=t=>b*Math.cos(rad(t)),res=t=>Math.abs(b*Math.sin(rad(t)));
    return {curves:[curve('Signed parameter bias',xs,bias),curve('Visible residual norm',xs,res),{name:'Selected direction',points:[[a,bias(a)],[a,res(a)]]}],
      result:'At '+a+'° the parameter bias is '+label(bias(a))+' and residual norm '+label(res(a))+'.'};
  });
  add('posterior-prediction',[['Parameter correlation',-.98,.98,.02,-.9],['Candidate sensor variance',.05,1,.05,.25]],
    'Measurement angle (degrees)','Remaining variance of normalised sum',([c,r])=>{
      const orig=1+c,post=t=>{const u=Math.cos(t*Math.PI/180),v=Math.sin(t*Math.PI/180);return orig-(1+c)**2*(u+v)**2/(2*(1+2*c*u*v+r));};
      const xs=grid(-90,90,180);
      return {curves:[curve('With observation',xs,post),curve('Before observation',xs,()=>orig)],
        result:'Target variance '+label(orig)+' initially, '+label(post(0))+' with a first-coordinate sensor, '+label(post(45))+' with a sensor along the target.'};
    });
  add('transient-growth',[['Coupling K',-20,20,.5,8]],'Dimensionless time','State / Euclidean norm',([k])=>{
    const xs=grid(0,5,200), y=t=>Math.exp(-2*t),x=t=>k*(Math.exp(-t)-y(t));
    return {curves:[curve('First component',xs,x),curve('Second component',xs,y),curve('State norm',xs,t=>Math.hypot(x(t),y(t)))],
      result:'At t = ln 2 the components are ('+label(k/4)+', 0.25). Largest eigenvalue of the energy-rate matrix '+label((-3+Math.hypot(1,k))/2)+'.'};
  });
  add('grazing-events',[['Threshold offset d',0,.025,.0005,.01],['Trajectory bias ε',-.01,.015,.0005,.001]],
    'Time (s)','Event function',([d,e])=>{
      const xs=grid(.78,1.22,176),g=t=>(t-1)**2-d, describe=z=>z< -1e-12?'no crossing':Math.abs(z)<1e-12?'tangent at 1 s':label(1-Math.sqrt(z))+' s and '+label(1+Math.sqrt(z))+' s';
      return {curves:[curve('Reference event function',xs,g),curve('Biased event function',xs,t=>g(t)+e),curve('Trigger level',xs,()=>0)],
        result:'Reference: '+describe(d)+'. Perturbed: '+describe(d-e)+'.'};
    });
  add('consistent-and-lumped-mass',[['Cells per selected wavelength',2,30,1,8]],'Cells per wavelength','Discrete / continuum frequency',([n])=>{
    const xs=grid(2,30,140),l=t=>Math.sin(Math.PI/t)/(Math.PI/t),c=t=>l(t)*Math.sqrt(3/(2+Math.cos(2*Math.PI/t)));
    return {curves:[curve('Lumped mass',xs,l),curve('Consistent mass',xs,c),{name:'Selected wavelength',points:[[n,l(n)],[n,c(n)]]}],
      result:'At '+n+' cells per wavelength the lumped ratio is '+label(l(n))+' and the consistent ratio '+label(c(n))+'.'};
  });
  add('conservative-remapping',[['Courant number C',.05,1,.05,.5],['Number of steps',1,40,1,24]],'Cell index','Cell average',([c,n])=>{
    const xs=grid(0,79,79);let u=xs.map(j=>j===12?1:0);
    for(let k=0;k<n;k++)u=u.map((v,j)=>(1-c)*v+c*u[(j+79)%80]);
    const mean=12+n*c;
    return {curves:[curve('Transported cell averages',xs,(_,j)=>u[j]),{name:'Exact centroid',points:[[mean,0],[mean,Math.max(...u)]]}],
      result:'Total cell sum '+label(u.reduce((s,v)=>s+v,0))+', centroid '+label(mean)+', added variance '+label(n*c*(1-c))+' cells².'};
  });
})();
