(() => {
  "use strict";
  const models=window.fundamentalStudies ||= {};
  const fmt=value=>Number(value.toPrecision(5)).toString();
  const axis=(start,end,n)=>Array.from({length:n+1},(_,j)=>start+(end-start)*j/n);
  const curve=(name,xs,f)=>({name,points:xs.map(x=>[x,f(x)])});
  const selected=(name,x,y)=>({name,points:[[x,y]]});
  const register=(key,controls,xLabel,yLabel,calculate)=>{
    models["math-lab-"+key]={controls,xLabel,yLabel,calculate};
  };

  register("range-sketch",[["Singular-value ratio σ₂/σ₁",.05,.95,.05,.45],["Power iterations",0,5,1,1]],
    "Starting probe angle to dominant right mode (degrees)","Spectral norm of the rank-one residual",([r,q])=>{
      const xs=axis(0,90,180);
      function residual(degrees,iterations){
        const angle=degrees*Math.PI/180;
        const x=degrees===90?0:Math.cos(angle), y=r**(2*iterations+1)*Math.sin(angle);
        return Math.hypot(y,r*x)/Math.hypot(x,y);
      }
      return {curves:[curve("Selected iteration count",xs,a=>residual(a,q)),curve("Single range probe",xs,a=>residual(a,0)),curve("Best possible rank-one residual",xs,()=>r)],
        result:"At 60°, residual "+fmt(residual(60,q))+" against the best possible "+fmt(r)+". A probe orthogonal to the leading mode gives residual one at every iteration count."};
    });

  register("network-gauge",[["Conductance between nodes 1 and 2 (S)",.1,4,.1,1],["Current from node 1 to node 3 (A)",.2,2,.1,1]],
    "Common voltage added to all nodes (V)","Potential or potential difference (V)",([g,current])=>{
      const xs=axis(-2,2,60),v1=current*(1+g)/(1+2*g),v2=current*g/(1+2*g);
      return {curves:[curve("Potential at node 1",xs,c=>v1+c),curve("Potential at node 2",xs,c=>v2+c),
        curve("Potential at node 3",xs,c=>c),curve("Measured drop from 1 to 2",xs,()=>v1-v2)],
        result:"Branch current I₁₂ = "+fmt(g*(v1-v2))+" A. Total Joule dissipation I(V₁−V₃) = "+fmt(current*v1)+" W at every gauge."};
    });

  register("stream-calibration",[["New observation coordinate",1,6,.1,4],["New observation's offset from y = x",-2,2,.1,1]],
    "Calibration input x","Measured or fitted output",([x,offset])=>{
      // Three existing observations are (-1,-1), (0,0), (1,1). The base normal matrix is diag(3,2).
      const v=1/3+x*x/2, intercept=offset/(3*(1+v)), slope=1+x*offset/(2*(1+v));
      const xs=axis(-1.5,6.4,90);
      return {curves:[curve("Before the update",xs,z=>z),curve("After the update",xs,z=>intercept+slope*z),
        {name:"Original observations",points:[[-1,-1],[0,0],[1,1]]},selected("New observation",x,x+offset)],
        result:"Updated slope "+fmt(slope)+", intercept "+fmt(intercept)+". New row leverage "+fmt(v/(1+v))+"; deleted prediction residual "+fmt(offset)+" versus training residual "+fmt(offset/(1+v))+"."};
    });

  register("nested-cosines",[["Intervals on the coarse cosine grid",4,24,1,8],["Chebyshev polynomial degree to inspect",0,80,1,32]],
    "Degree of the input Chebyshev polynomial","Integral over [−1,1]",([n,m])=>{
      const moment=k=>k%2?0:2/(1-k*k);
      function interpolated(k,intervals){const folded=k%(2*intervals);return moment(Math.min(folded,2*intervals-folded));}
      const xs=axis(0,80,80);
      return {curves:[curve("Exact polynomial moment",xs,moment),curve("Coarse cosine grid",xs,k=>interpolated(k,n)),
        curve("Twice as many intervals",xs,k=>interpolated(k,2*n)),selected("Chosen degree on coarse grid",m,interpolated(m,n))],
        result:"Degree "+m+" integrates to "+fmt(moment(m))+". The two nested estimates are "+fmt(interpolated(m,n))+" and "+fmt(interpolated(m,2*n))+"."};
    });

  register("filon-carrier",[["Uniform integration panels",2,24,1,6],["Selected carrier (cycles in the interval)",0,12,.1,6]],
    "Carrier cycles on [0,1]","Cosine transform of exp(x)",([n,freq])=>{
      const h=.5/n, xs=axis(0,12,240), exact=f=>{const w=2*Math.PI*f;return (Math.E*(Math.cos(w)+w*Math.sin(w))-1)/(1+w*w);};
      function panelSum(f,filon){
        const w=2*Math.PI*f,z=w*h, sinc=z===0?1:Math.sin(z)/z;
        const first=Math.abs(z)<.02?z/3-z**3/30+z**5/840:(Math.sin(z)-z*Math.cos(z))/(z*z);
        let sum=0;
        for(let j=0;j<n;j++){
          const left=j/n,right=(j+1)/n,c=(left+right)/2,fl=Math.exp(left),fr=Math.exp(right);
          if(filon)sum+=h*((fl+fr)*sinc*Math.cos(w*c)-(fr-fl)*first*Math.sin(w*c));
          else sum+=h*(fl*Math.cos(w*left)+fr*Math.cos(w*right));
        }
        return sum;
      }
      return {curves:[curve("Closed integral",xs,exact),curve("Linear-envelope Filon",xs,f=>panelSum(f,true)),
        curve("Direct trapezoid",xs,f=>panelSum(f,false)),selected("Selected exact response",freq,exact(freq))],
        result:"At the selected frequency the exact response is "+fmt(exact(freq))+". Filon gives "+fmt(panelSum(freq,true))+" and trapezoids give "+fmt(panelSum(freq,false))+"."};
    });

  register("control-variate",[["Coefficient β in X² − β(X − ½)",-1,3,.05,1],["Sample count to inspect",8,512,8,64]],
    "Independent uniform observations","Standard error of the estimated mean",([beta,n])=>{
      const xs=axis(4,512,254),variance=1/180+(beta-1)**2/12,raw=4/45;
      return {curves:[curve("Direct estimator",xs,k=>Math.sqrt(raw/k)),curve("With selected control",xs,k=>Math.sqrt(variance/k)),
        curve("Optimal coefficient",xs,k=>Math.sqrt(1/(180*k))),selected("Selected budget",n,Math.sqrt(variance/n))],
        result:"Variance ratio controlled/direct = "+fmt(variance/raw)+". With "+n+" independent samples the controlled standard error is "+fmt(Math.sqrt(variance/n))+"."};
    });

  register("validation-leverage",[["Sensitivity of the fifth sensor",.25,5,.25,3],["Fifth reading offset",-2,2,.1,.8]],
    "Base-ten logarithm of ridge penalty λ","Mean squared prediction score",([gain,offset])=>{
      const x=[1,1,1,1,gain], y=[1.1,.9,1.05,.95,gain+offset], norm=4+gain*gain, rhs=x.reduce((s,a,i)=>s+a*y[i],0);
      function scores(log){
        const den=norm+10**log, residual=y.map((v,i)=>v-x[i]*rhs/den);
        const loo=residual.reduce((s,r,i)=>s+(r/(1-x[i]*x[i]/den))**2,0)/x.length;
        const mse=residual.reduce((s,r)=>s+r*r,0)/x.length;
        return [loo,mse/(1-norm/(x.length*den))**2];
      }
      const xs=axis(-2,1.3,100), one=scores(0);
      return {curves:[curve("Explicit leave-one-out prediction identity",xs,t=>scores(t)[0]),curve("Trace-based GCV",xs,t=>scores(t)[1])],
        result:"At λ = 1, the fifth sensor has leverage "+fmt(gain*gain/(norm+1))+". The leave-one-out and generalised scores are "+fmt(one[0])+" and "+fmt(one[1])+"."};
    });

  register("sparse-bias",[["Penalty λ",0,2,.05,.4],["Multiplier for measurement perturbations",0,2,.05,1]],
    "Orthogonal response channel","Coefficient amplitude",([penalty,noise])=>{
      const truth=[2.4,-1.7,.35,0,0,0],error=[.1,.18,-.12,.25,-.18,.08],xs=axis(1,6,5);
      const y=truth.map((v,i)=>v+noise*error[i]), estimate=y.map(v=>Math.sign(v)*Math.max(0,Math.abs(v)-penalty));
      return {curves:[curve("Underlying coefficients",xs,i=>truth[i-1]),curve("Recorded channels",xs,i=>y[i-1]),curve("Penalised estimate",xs,i=>estimate[i-1])],
        result:"Recovered "+estimate.filter(v=>v!==0).length+" nonzero coefficients of six. Euclidean reconstruction error "+fmt(Math.hypot(...estimate.map((v,i)=>v-truth[i])))+"."};
    });

  register("repeated-readings",[["Prior parameter correlation",0,.98,.02,.8],["Shared calibration standard deviation",0,1,.05,.4]],
    "Repeated readings of the same first channel","Posterior standard deviation (prior units)",([rho,bias])=>{
      const xs=axis(1,100,99), variance=(n,shared)=>shared?bias*bias+.16/n:(bias*bias+.16)/n;
      const sd=(n,corr,shared)=>Math.sqrt(1-corr*corr/(1+variance(n,shared)));
      return {curves:[curve("Observed parameter, shared calibration",xs,n=>sd(n,1,true)),curve("Unobserved parameter, shared calibration",xs,n=>sd(n,rho,true)),
        curve("Observed parameter, independent-noise assumption",xs,n=>sd(n,1,false))],
        result:"After 100 observations the true posterior widths are "+fmt(sd(100,1,true))+" and "+fmt(sd(100,rho,true))+". Single-reading noise standard deviation "+fmt(Math.hypot(.4,bias))+"."};
    });
  models["math-lab-network-gauge"].localModel={file:"networks-and-calibration.py",
    args:([g,i])=>"network --conductance "+g+" --current "+i};
  models["math-lab-stream-calibration"].localModel={file:"networks-and-calibration.py",
    args:([x,d])=>"line --coordinate "+x+" --offset "+d};
  models["math-lab-validation-leverage"].localModel={file:"networks-and-calibration.py",
    args:([a,d])=>"validation --sensitivity "+a+" --offset "+d};
})();
