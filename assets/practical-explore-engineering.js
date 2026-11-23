// Browser calculations for the engineering extensions; companion notebooks run OneSimulation.
(() => {
  'use strict';
  const studies = window.fundamentalStudies ||= {};
  const pi = Math.PI;
  const curve = (name, start, end, fn, count=160) => ({name, points:Array.from({length:count+1},(_,i)=>{
    const x=start+(end-start)*i/count; return [x,fn(x)];
  })});
  const norm = a => Math.hypot(...a);
  function solve3(matrix, rhs) {
    const a=matrix.map((row,i)=>[...row,rhs[i]]);
    for(let k=0;k<3;k++) {
      let pivot=k;
      for(let i=k+1;i<3;i++) if(Math.abs(a[i][k])>Math.abs(a[pivot][k])) pivot=i;
      [a[k],a[pivot]]=[a[pivot],a[k]];
      if(Math.abs(a[k][k])<1e-14) throw new Error('Singular equilibrium matrix');
      const d=a[k][k]; for(let j=k;j<4;j++) a[k][j]/=d;
      for(let i=0;i<3;i++) if(i!==k){const m=a[i][k];for(let j=k;j<4;j++) a[i][j]-=m*a[k][j];}
    }
    return a.map(row=>row[3]);
  }

  studies['explore-engineering-tripod'] = {
    controls:[['Apex height (m)',0.3,2,0.05,1],['Foot radius (m)',0.4,2,0.05,1],['Downward load (kN)',1,10,0.5,6],['Horizontal load (kN)',0,12,0.5,3],['Horizontal load azimuth (degrees)',0,360,5,0]],
    xLabel:'Horizontal load (kN)', yLabel:'Member force: positive tension (kN)',
    calculate([h,r,w,horizontal,azimuth]) {
      const alpha=azimuth*pi/180, apex=[0,0,h];
      const feet=[0,120,240].map(angle=>[r*Math.cos(angle*pi/180),r*Math.sin(angle*pi/180),0]);
      const length=Math.hypot(r,h), directions=feet.map(p=>[p[0]/length,p[1]/length,-h/length]);
      const matrix=[0,1,2].map(i=>directions.map(p=>p[i]));
      const forces=load=>solve3(matrix,[-load*Math.cos(alpha),-load*Math.sin(alpha),w]);
      const selected=forces(horizontal), external=[horizontal*Math.cos(alpha),horizontal*Math.sin(alpha),-w];
      const residual=norm(external.map((f,i)=>f+selected.reduce((s,n,j)=>s+n*directions[j][i],0)));
      const color=n=>Math.abs(n)<1e-9?'#717b7c':n<0?'#b34b28':'#176e74';
      const lines=feet.map((p,i)=>({points:[apex,p],color:color(selected[i]),width:3+Math.min(5,Math.abs(selected[i])/3)}));
      const points=[{position:apex,label:'Loaded apex',color:'#283a3c',radius:5}];
      feet.forEach((p,i)=>points.push({position:p,label:'N'+(i+1)+' = '+selected[i].toFixed(2)+' kN',color:color(selected[i]),radius:5}));
      const arrowScale=r/12;
      const end=[arrowScale*external[0],arrowScale*external[1],h+arrowScale*external[2]];
      lines.push({points:[apex,end],color:'#694a85',width:3});
      points.push({position:end,label:'Applied resultant',color:'#694a85',radius:4});
      return {curves:[0,1,2].map(i=>curve('Member '+(i+1),0,12,load=>forces(load)[i])),
        result:'At the selected load: '+selected.map((n,i)=>'N'+(i+1)+' = '+n.toFixed(3)+' kN').join('; ')+'. Joint force residual '+residual.toExponential(1)+' kN. '+(selected.some(n=>n>1e-8)?'Tension requires anchored feet.':'All members are in compression or unloaded.'),
        scene:{lines,points,faces:[{vertices:feet,color:'rgba(118,147,139,0.16)'}],axisLabels:['x (m)','y (m)','z (m)'],caption:'Three-dimensional pin-jointed tripod at the selected load. Orange members carry compression; teal members carry tension; grey means unloaded. The purple segment shows the applied force direction with a symbolic length. Members retain their undeformed geometry.'}};
    }
  };

  studies['explore-engineering-beam'] = {
    controls:[['Mode number',1,3,1,1],['Beam length (m)',0.5,2,0.05,1],['Beam depth (mm)',5,25,1,10],['Tip amplitude (mm)',5,50,1,30],['Oscillation phase (degrees)',0,360,5,0]],
    xLabel:'Position along beam (m)', yLabel:'Transverse displacement (mm)',
    calculate([mode,length,depthMM,amplitude,phase]) {
      const beta=[1.875104068711961,4.694091132974175,7.854757438237612][mode-1], depth=depthMM/1000;
      const sigma=(Math.cosh(beta)+Math.cos(beta))/(Math.sinh(beta)+Math.sin(beta));
      const shape=s=>Math.cosh(beta*s)-Math.cos(beta*s)-sigma*(Math.sinh(beta*s)-Math.sin(beta*s));
      const phi=s=>shape(s)/shape(1), instant=Math.cos(phase*pi/180);
      const deflection=x=>amplitude*phi(x/length)*instant;
      const frequency=beta*beta*Math.sqrt(210e9*depth*depth/(12*7800))/(2*pi*length*length);
      const lines=[],faces=[],n=48,width=0.03;
      const position=(i,side,vertical)=>{const x=length*i/n;return[x,side*width/2,deflection(x)/1000+vertical*depth/2];};
      for(const side of [-1,1]) for(const vertical of [-1,1]) lines.push({points:Array.from({length:n+1},(_,i)=>position(i,side,vertical)),color:'#176e74',width:2});
      for(let i=0;i<n;i++) for(const side of [-1,1]) faces.push({vertices:[position(i,side,-1),position(i+1,side,-1),position(i+1,side,1),position(i,side,1)],color:'rgba(39,120,128,0.24)'});
      for(let i=0;i<n;i++) faces.push({vertices:[position(i,-1,1),position(i+1,-1,1),position(i+1,1,1),position(i,1,1)],color:'rgba(39,120,128,0.32)'});
      lines.push({points:[[0,0,0],[length,0,0]],color:'#818b89',width:1});
      for(const i of [0,n]) lines.push({points:[position(i,-1,-1),position(i,1,-1),position(i,1,1),position(i,-1,1),position(i,-1,-1)],color:'#283a3c',width:2});
      return {curves:[curve('Selected instant',0,length,deflection),curve('Positive modal envelope',0,length,x=>amplitude*phi(x/length)),curve('Negative modal envelope',0,length,x=>-amplitude*phi(x/length))],
        result:'Mode '+mode+': '+frequency.toFixed(4)+' Hz; period '+(1000/frequency).toFixed(3)+' ms. The analytical Euler–Bernoulli mode has a clamped root and a free tip. Shape amplitude is an illustration of the linear mode, not a predicted forced response.',
        scene:{lines,faces,points:[{position:[0,0,0],label:'Clamp',color:'#283a3c',radius:4}],axisLabels:['x (m)','y (m)','z (m)'],caption:'A three-dimensional extrusion of the analytical beam mode. Width is 30 mm; depth and modal phase are adjustable. The section is drawn without the small rotation correction. Frequency and centreline follow the slender-beam model; the companion notebook checks the numerical eigenproblem.'}};
    }
  };

  function equilibrium(air, pressure) {
    const hf=[-393520,-110530,-241830,0,0,0], cp=[51,33,42,31,35,33], entropy=[213.79,197.66,188.84,130.68,205.15,191.61];
    const reactions=[[-1,1,0,0,0.5,0],[0,0,-1,1,0.5,0]], R=8.314462618;
    const logistic=z=>1/(1+Math.exp(-z));
    function state(z){return [logistic(z[0]),2*logistic(z[1]),1000*z[2]];}
    function residual(z) {
      const [x,y,t]=state(z), moles=[1-x,x,2-y,y,2*(air-1)+(x+y)/2,7.52*air];
      const total=moles.reduce((a,b)=>a+b,0);
      const enthalpy=hf.map((h,i)=>h+cp[i]*(t-298.15));
      const gibbs=enthalpy.map((h,i)=>h-t*(entropy[i]+cp[i]*Math.log(t/298.15)));
      const chemical=reactions.map(nu=>nu.reduce((s,n,i)=>s+n*(Math.log(pressure*moles[i]/total)+gibbs[i]/(R*t)),0));
      return [...chemical,(moles.reduce((sum,n,i)=>sum+n*enthalpy[i],0)+74870)/1e5];
    }
    let z=[Math.log(0.06/0.94),Math.log(0.02/0.98),2.2], r=residual(z);
    for(let iteration=0;iteration<24 && norm(r)>1e-10;iteration++) {
      const step=1e-5, columns=z.map((_,j)=>{const test=[...z];test[j]+=step;return residual(test).map((v,i)=>(v-r[i])/step);});
      const change=solve3([0,1,2].map(i=>columns.map(column=>column[i])),r.map(v=>-v));
      let accepted=false;
      for(let damping=1;damping>1/4096;damping/=2) {
        const candidate=z.map((v,i)=>v+damping*change[i]);
        if(candidate[2]<0.5 || candidate[2]>3.5) continue;
        const next=residual(candidate);
        if(norm(next)<norm(r)){z=candidate;r=next;accepted=true;break;}
      }
      if(!accepted) break;
    }
    if(norm(r)>1e-7) throw new Error('Equilibrium residual exceeded tolerance');
    return {state:state(z), residual:norm(r)};
  }
  studies['explore-engineering-reacting-mixture'] = {
    controls:[['Air ratio',1,1.4,0.01,1],['Total pressure (bar)',1,10,0.5,1]],
    xLabel:'Air ratio', yLabel:'Adiabatic temperature (K)',
    calculate([air,pressure]) {
      const selected=equilibrium(air,pressure), [x,y,t]=selected.state;
      const complete=a=>298.15+802310/(51+2*42+2*(a-1)*35+7.52*a*33);
      return {curves:[curve('Six species at '+pressure+' bar',1,1.4,a=>equilibrium(a,pressure).state[2],50),curve('Six species at 1 bar',1,1.4,a=>equilibrium(a,1).state[2],50),curve('Complete conversion',1,1.4,complete,50),{name:'Selected operating point',points:[[air,t]]}],
        result:'Selected state: '+t.toFixed(2)+' K; CO extent '+x.toFixed(5)+' mol; H₂ extent '+y.toFixed(5)+' mol per mole of methane. Dissociation lowers the complete-conversion estimate by '+(complete(air)-t).toFixed(2)+' K. Scaled equilibrium/energy residual '+selected.residual.toExponential(1)+'. Constant heat capacities and six ideal-gas species, as in the notebook.'};
    }
  };

  studies['explore-engineering-backwater'] = {
    controls:[['Discharge (m³/s)',2,10,0.5,6],['Manning roughness (s/m⅓)',0.015,0.045,0.001,0.030],['Outlet depth (m)',1.6,3.2,0.05,2.2]],
    xLabel:'Distance upstream from outlet (m)', yLabel:'Water depth (m)',
    calculate([q,roughness,outlet]) {
      const width=4, slope=0.001, g=9.81;
      const friction=y=>(roughness*q/(width*y*(width*y/(width+2*y))**(2/3)))**2;
      const critical=Math.cbrt(q*q/(g*width*width));
      let lower=critical,upper=12;
      for(let i=0;i<70;i++){const middle=(lower+upper)/2;if(friction(middle)>slope) lower=middle;else upper=middle;}
      const normal=(lower+upper)/2;
      // u = -x increases upstream, so the downstream differential equation changes sign.
      const f=y=>(friction(y)-slope)/(1-q*q/(g*width*width*y*y*y));
      const points=[[0,outlet]];let y=outlet;const step=2;
      for(let u=step;u<=1000;u+=step){const k1=f(y),k2=f(y+step*k1/2),k3=f(y+step*k2/2),k4=f(y+step*k3);y+=step*(k1+2*k2+2*k3+k4)/6;points.push([u,y]);}
      return {curves:[{name:'Selected downstream control',points},curve('Normal depth',0,1000,()=>normal,2),curve('Critical depth',0,1000,()=>critical,2)],
        result:'At 1 km upstream: '+y.toFixed(3)+' m. Normal depth '+normal.toFixed(3)+' m; critical depth '+critical.toFixed(3)+' m. Outlet Froude number '+Math.sqrt(q*q/(g*width*width*outlet**3)).toFixed(3)+'. Fixed 4 m width and 0.001 bed slope; upstream RK4 steps are 2 m. The selected range stays subcritical.'};
    }
  };

  studies['explore-engineering-fatigue'] = {
    controls:[['Surface factor',0.65,0.95,0.01,0.8],['Elastic concentration Kt',1,2.5,0.05,1.65],['Nominal mean stress (MPa)',0,220,5,30],['Nominal alternating stress (MPa)',40,220,5,100]],
    xLabel:'Nominal tensile mean stress (MPa)', yLabel:'Nominal alternating stress (MPa)',
    calculate([surface,kt,mean,amplitude]) {
      const strength=280*surface*0.88, kf=1+0.8*(kt-1), ultimate=650;
      const allowable=m=>strength*(1-kt*m/ultimate)/kf;
      const equivalent=kf*amplitude/(1-kt*mean/ultimate),life=1e6*(strength/equivalent)**5;
      return {curves:[curve('One-million-cycle Goodman boundary',0,220,allowable),curve('Same component without a notch',0,220,m=>strength*(1-m/ultimate)),{name:'Selected nominal cycle',points:[[mean,amplitude]]}],
        result:'Kf = '+kf.toFixed(3)+', corrected strength '+strength.toFixed(2)+' MPa, equivalent reversed amplitude '+equivalent.toFixed(2)+' MPa. Illustrative constant-amplitude life: '+life.toExponential(3)+' cycles; damage per cycle '+(1/life).toExponential(3)+'. This finite-life model has no endurance knee and does not check yielding.'};
    }
  };

  studies['explore-engineering-cascade'] = {
    controls:[['Achieved inner bandwidth (rad/s)',50,250,5,100],['Outer-path delay (ms)',0,20,0.5,6],['Chosen outer crossover (rad/s)',5,90,1,50]],
    xLabel:'Outer gain crossover (rad/s)', yLabel:'Phase margin (degrees)',
    calculate([inner,delayMS,crossover]) {
      const delay=delayMS/1000, margin=w=>90-(Math.atan(w/inner)+w*delay)*180/pi;
      const gain=crossover*Math.sqrt(1+(crossover/inner)**2);
      let lower=0,upper=1000;
      for(let i=0;i<60;i++){const middle=(lower+upper)/2;if(margin(middle)>45) lower=middle;else upper=middle;}
      const limit=(lower+upper)/2,discriminant=inner*inner-4*inner*gain;
      const decay=discriminant<0?inner/2:(inner-Math.sqrt(discriminant))/2;
      return {curves:[curve('Selected delay',1,100,margin),curve('Zero-delay reference',1,100,w=>90-Math.atan(w/inner)*180/pi),curve('45° requirement',1,100,()=>45,2),{name:'Chosen crossover',points:[[crossover,margin(crossover)]]}],
        result:'Chosen gain K = '+gain.toFixed(3)+' /s gives '+margin(crossover).toFixed(2)+'° phase margin. The 45° boundary is '+limit.toFixed(2)+' rad/s. The delay-free closed-loop pole margin is '+decay.toFixed(3)+' /s; it is a separate diagnostic from the delayed phase margin.'};
    }
  };
})();
