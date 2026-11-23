(() => {
  'use strict';
  const studies = window.fundamentalStudies ||= {};
  const pi = Math.PI;
  function curve(name,start,end,fn,n=180){
    return {name,points:Array.from({length:n+1},(_,i)=>{
      const x=start+(end-start)*i/n;
      return [x,fn(x)];
    })};
  }
  studies['engineering-methods-cell-torsion'] = {
    controls:[['Left cell width (mm)',25,85,1,60],['Web thickness (mm)',0.4,3,0.1,1],['Torque (N m)',3,25,1,10]],
    xLabel:'Right cell width (mm)',yLabel:'Wall shear flow (kN/m)',
    calculate([left,web,torque]){
      function flow(right){
        const h=0.040,w1=left/1000,w2=right/1000,A=h*w1,B=h*w2;
        const c1=(2*w1+h)/0.001,c2=(2*w2+h)/0.001,cw=h/(web/1000);
        const ratio=(B*(c1+cw)+A*cw)/(A*(c2+cw)+B*cw);
        const q1=torque/(2*(A+B*ratio));
        return [q1/1000,ratio*q1/1000,(1-ratio)*q1/1000];
      }
      const q=flow(40);
      return {curves:[curve('Left outer walls',25,85,x=>flow(x)[0]),curve('Right outer walls',25,85,x=>flow(x)[1]),curve('Signed web difference',25,85,x=>flow(x)[2])],
        result:'At 40 mm right width the two circulations are '+q[0].toFixed(3)+' and '+q[1].toFixed(3)+' kN/m. The shared web carries their signed difference, '+q[2].toFixed(3)+' kN/m.'};
    }
  };
  studies['engineering-methods-intercooling'] = {
    controls:[['Overall pressure ratio',4,16,1,9],['Second inlet temperature (K)',295,355,5,300],['Second efficiency',0.65,0.95,0.01,0.85]],
    xLabel:'First-stage pressure ratio',yLabel:'Specific shaft work (kJ/kg)',
    calculate([ratio,T2,eta2]){
      const T1=300,eta1=0.85,a=2/7,cp=1.005;
      const work=r=>cp*(T1/eta1*(r**a-1)+T2/eta2*((ratio/r)**a-1));
      const optimal=Math.max(1,Math.min(ratio,Math.sqrt(ratio*(T2*eta1/(T1*eta2))**(1/a))));
      return {curves:[curve('Two stages',1,ratio,work),curve('Single stage at first efficiency',1,ratio,()=>cp*T1/eta1*(ratio**a-1))],
        result:'Minimum work is '+work(optimal).toFixed(2)+' kJ/kg at first ratio '+optimal.toFixed(3)+'. The equal-ratio split requires '+work(Math.sqrt(ratio)).toFixed(2)+' kJ/kg.'};
    }
  };
  studies['engineering-methods-interface'] = {
    controls:[['Relative flow speed (m/s)',2,12,0.1,7],['Surface tension (mN/m)',30,85,1,72]],
    xLabel:'Wavelength (mm)',yLabel:'Exponential growth rate (1/s)',
    calculate([speed,tension]){
      const a=1000,b=1.2,sg=tension/1000,g=9.81;
      const kstar=Math.sqrt((a-b)*g/sg);
      const threshold=Math.sqrt((a+b)/(a*b)*2*Math.sqrt((a-b)*g*sg));
      const growth=(l,V)=>{const k=2*pi/(l/1000);return Math.sqrt(Math.max(0,a*b/(a+b)**2*(k*V)**2-((a-b)*g*k+sg*k**3)/(a+b)));};
      return {curves:[curve('Selected flow',4,75,l=>growth(l,speed),400),curve('At 90% of selected speed',4,75,l=>growth(l,0.9*speed),400)],
        result:'The ideal minimum threshold is '+threshold.toFixed(2)+' m/s at wavelength '+(2000*pi/kstar).toFixed(2)+' mm. At that wavelength selected growth is '+growth(2000*pi/kstar,speed).toFixed(1)+' per second.'};
    }
  };
  studies['engineering-methods-whirl'] = {
    controls:[['Eccentricity (µm)',5,80,1,20],['Damping ratio',0.02,0.20,0.01,0.05]],
    xLabel:'Spin / undamped natural frequency',yLabel:'Synchronous orbit radius (µm)',
    calculate([e,z]){
      const den=r=>Math.hypot(1-r*r,2*z*r);
      return {curves:[curve('Shaft centre',0,2.4,r=>e*r*r/den(r),400),curve('Mass centre',0,2.4,r=>e*Math.hypot(1,2*z*r)/den(r),400)],
        result:'At unit speed ratio the shaft-centre radius is '+(e/(2*z)).toFixed(1)+' µm and its lag behind unbalance is 90°. The mass centre and shaft centre are different points.'};
    }
  };
  studies['engineering-methods-cam'] = {
    controls:[['Lift (mm)',4,18,1,10],['Rise angle (°)',60,180,5,120],['Prime radius (mm)',15,50,1,25]],
    xLabel:'Cam angle during rise (°)',yLabel:'Pressure angle magnitude (°)',
    calculate([lift,span,prime]){
      const beta=span*pi/180;
      function angle(deg,cycloidal){
        const u=deg/span;
        const s=cycloidal?lift*(u-Math.sin(2*pi*u)/(2*pi)):lift*(1-Math.cos(pi*u))/2;
        const slope=cycloidal?lift*(1-Math.cos(2*pi*u))/beta:lift*pi*Math.sin(pi*u)/(2*beta);
        return 180/pi*Math.atan(slope/(s+prime));
      }
      const cycloid=curve('Cycloidal rise',0,span,x=>angle(x,true),360);
      const harmonic=curve('Simple harmonic rise',0,span,x=>angle(x,false),360);
      const peak=Math.max(...cycloid.points.map(p=>p[1]));
      return {curves:[cycloid,harmonic],result:'The cycloidal peak pressure angle is '+peak.toFixed(2)+'°. At half rise it is '+angle(span/2,true).toFixed(2)+'°; the harmonic comparison has an acceleration jump at a dwell.'};
    }
  };
  studies['engineering-methods-observer'] = {
    controls:[['Observer natural frequency (Hz)',0.5,6,0.1,2],['Sinusoidal sensor error (mm)',0,0.10,0.005,0.03],['Sensor error frequency (Hz)',3,16,0.5,10]],
    xLabel:'Time (s)',yLabel:'Estimated or true carriage velocity (mm/s)',
    calculate([frequency,noise,sensor]){
      const w=2*pi*frequency,L1=2*0.75*w,L2=w*w,dt=0.0002;
      const state=[[0,0],[0,0]],traces=[[],[],[]],trueV=0.4;
      for(let i=0;i<=10000;i++){
        const t=i*dt,position=1+trueV*t;
        if(i%25===0){traces[0].push([t,state[0][1]]);traces[1].push([t,state[1][1]]);traces[2].push([t,trueV]);}
        for(let j=0;j<2;j++){
          const [x,v]=state[j],y=position+(j?noise*Math.sin(2*pi*sensor*t):0),r=y-x;
          state[j]=[x+dt*(v+L1*r),v+dt*L2*r];
        }
      }
      const omega=2*pi*sensor;
      const ripple=noise*L2*omega/Math.hypot(w*w-omega*omega,L1*omega);
      return {curves:[{name:'Clean measurement',points:traces[0]},{name:'Sensor error included',points:traces[1]},{name:'True velocity',points:traces[2]}],
        result:'The predicted steady velocity ripple amplitude from the sensor error is '+ripple.toFixed(3)+' mm/s. Initial true position is 1 mm; both estimated initial states are zero.'};
    }
  };
  studies['engineering-methods-hertz-line'] = {
    controls: [['Line load (kN/m)', 5, 60, 1, 20], ['Reduced radius (mm)', 3, 20, 1, 10], ['Reduced modulus (GPa)', 50, 180, 5, 115]],
    xLabel: 'Position across contact (µm)', yLabel: 'Normal pressure (MPa)',
    calculate([load, radius, modulus]) {
      const halfWidth = Math.sqrt(4 * load * 1000 * radius / 1000 / (pi * modulus * 1e9));
      const pressure = (x, factor) => {
        const a = halfWidth * Math.sqrt(factor);
        return 2 * factor * load * 1000 / (pi * a * 1e6) * Math.sqrt(Math.max(0, 1 - (x * 1e-6 / a) ** 2));
      };
      const span = halfWidth * 1.6e6;
      return {curves: [curve('Selected load', -span, span, x => pressure(x, 1)), curve('Double load', -span, span, x => pressure(x, 2))],
        result: 'At the selected load the half-width is '+(halfWidth*1e6).toFixed(1)+' µm and the peak pressure is '+pressure(0, 1).toFixed(1)+' MPa.'};
    }
  };
  studies['engineering-methods-springback'] = {
    controls: [['Applied curvature / first-yield curvature', 1, 4, 0.05, 2], ['Yield stress (MPa)', 100, 400, 10, 250], ['Half-thickness (mm)', 1, 5, 0.25, 2]],
    xLabel: 'Distance from middle surface (mm)', yLabel: 'Longitudinal stress (MPa)',
    calculate([ratio, yieldStress, half]) {
      const load = y => yieldStress * Math.max(-1, Math.min(1, ratio * y / half));
      const normalizedMoment = (3 - 1 / ratio ** 2) / 2;
      const residual = y => load(y) - yieldStress * normalizedMoment * y / half;
      const curvature = (ratio - normalizedMoment) * yieldStress * 1e6 / (200e9 * half / 1000);
      return {curves: [curve('Under load', -half, half, load), curve('After elastic unloading', -half, half, residual)],
        result: 'Residual curvature is '+curvature.toFixed(3)+' per metre. The previously tensile surface has residual stress '+residual(half).toFixed(2)+' MPa.'};
    }
  };
  studies['engineering-methods-shear-centre'] = {
    controls: [['Midline web height (mm)', 35, 120, 1, 60], ['Flange width (mm)', 12, 55, 1, 30], ['Vertical load (N)', 25, 200, 5, 100]],
    xLabel: 'Applied load position from web (mm)', yLabel: 'Signed twisting moment (N m)',
    calculate([height, width, load]) {
      const offset = 3 * width ** 2 / (height + 6 * width);
      const centroid = width ** 2 / (height + 2 * width);
      const moment = x => load * (x + offset) / 1000;
      return {curves: [curve('About the shear centre', -1.5*offset, width, moment), curve('Moment about the web line', -1.5*offset, width, x => load*x/1000)],
        result: 'The zero-twist line lies '+offset.toFixed(2)+' mm behind the web. A centroidal load produces '+moment(centroid).toFixed(3)+' N m of torque.'};
    }
  };
  studies['engineering-methods-condensate-film'] = {
    controls: [['Wall subcooling (K)', 2, 12, 0.5, 5], ['Plate length (m)', 0.1, 0.5, 0.01, 0.20]],
    xLabel: 'Distance from upper edge (m)', yLabel: 'Heat-transfer coefficient (kW/m² K)',
    calculate([deltaT, length]) {
      const thickness = x => (4*2.9e-4*0.68*deltaT*x/(960*(960-0.6)*9.81*2.26e6))**0.25;
      const local = x => 0.68 / thickness(x) / 1000;
      const mean = x => 4*local(x)/3;
      return {curves:[curve('Local value', length/80, length, local),curve('Mean from upper edge', length/80, length, mean)],
        result:'The outlet film is '+(thickness(length)*1e6).toFixed(1)+' µm thick. A 0.10 m wide plate rejects '+(mean(length)*deltaT*length*0.1).toFixed(3)+' kW.'};
    }
  };
  studies['engineering-methods-radiation-shield'] = {
    controls:[['Hot surface temperature (K)',400,650,5,500],['Cold surface temperature (K)',250,350,5,300],['Shield emissivity, both faces',0.05,0.9,0.05,0.15]],
    xLabel:'Emissivity of both external plates', yLabel:'Net radiative heat flux (W/m²)',
    calculate([hot,cold,shield]) {
      const difference = 5.670374419e-8 * (hot**4-cold**4);
      const bare = e => difference/(2/e-1);
      const protectedFlux = e => difference/(2/e+2/shield-2);
      return {curves:[curve('Bare gap',0.1,1,bare),curve('One floating shield',0.1,1,protectedFlux)],
        result:'At plate emissivity 0.8 the bare and shielded fluxes are '+bare(0.8).toFixed(1)+' and '+protectedFlux(0.8).toFixed(1)+' W/m². Both faces of the isothermal shield share temperature '+((hot**4/2+cold**4/2)**0.25).toFixed(1)+' K.'};
    }
  };
  studies['engineering-methods-thermal-penetration'] = {
    controls:[['Drive frequency (Hz)',0.05,0.5,0.01,0.10],['Thermal diffusivity (mm²/s)',2,20,0.5,10],['Sensor separation (mm)',1,8,0.25,3]],
    xLabel:'Time (s)', yLabel:'Temperature excess (K)',
    calculate([frequency,alpha,depth]) {
      const delta = Math.sqrt(alpha/(pi*frequency));
      const ratio = Math.exp(-depth/delta);
      const temperature = (time,x) => 5*Math.exp(-x/delta)*Math.cos(2*pi*frequency*time-x/delta);
      return {curves:[curve('At surface',0,2/frequency,t=>temperature(t,0)),curve('At sensor depth',0,2/frequency,t=>temperature(t,depth))],
        result:'Penetration depth is '+delta.toFixed(2)+' mm. At the selected depth the amplitude ratio is '+ratio.toFixed(3)+' and the phase lag is '+(depth/delta*180/pi).toFixed(1)+'°.'};
    }
  };
  studies['engineering-methods-sedimentation'] = {
    controls:[['Particle radius (µm)',3,30,1,10],['Liquid viscosity (mPa s)',1,15,0.5,4],['Density excess (kg/m³)',100,1400,50,500]],
    xLabel:'Particle radius (µm)', yLabel:'Terminal settling speed (mm/s)',
    calculate([radius,viscosity,excess]) {
      const speed = (r,mu) => 2*excess*9.81*(r*1e-6)**2/(9*mu*1e-3)*1000;
      const value=speed(radius,viscosity);
      const reynolds=1000*2*radius*1e-6*value*1e-3/(viscosity*1e-3);
      return {curves:[curve('Selected viscosity',3,30,r=>speed(r,viscosity)),curve('Twice the viscosity',3,30,r=>speed(r,2*viscosity))],
        result:'For the selected radius the Stokes prediction is '+value.toFixed(4)+' mm/s, giving diameter Reynolds number '+reynolds.toPrecision(3)+'. Compare this number with unity before trusting the approximation.'};
    }
  };
  studies['engineering-methods-taylor-aris'] = {
    controls:[['Tube radius (µm)',40,140,5,100],['Mean velocity (mm/s)',0.1,1.2,0.05,0.5],['Molecular diffusivity (nm²/ns)',0.5,2,0.1,1]],
    xLabel:'Elapsed time (s)', yLabel:'Axial standard deviation (mm)',
    calculate([radius,speed,molecular]) {
      const a=radius*1e-6, u=speed*1e-3, diffusivity=molecular*1e-9;
      const mixing=a*a/diffusivity, effective=diffusivity+(u*a)**2/(48*diffusivity);
      const width=(t,D)=>1000*Math.sqrt(2*D*t);
      return {curves:[curve('Shear and radial diffusion',5*mixing,15*mixing,t=>width(t,effective)),curve('Molecular axial diffusion',5*mixing,15*mixing,t=>width(t,diffusivity))],
        result:'Radial mixing scale is '+mixing.toFixed(2)+' s and effective diffusivity is '+effective.toExponential(3)+' m²/s. Ten mixing times after injection the effective axial rms width is '+width(10*mixing,effective).toFixed(2)+' mm.'};
    }
  };
  studies['engineering-methods-momentum-layer'] = {
    controls:[['Outer speed (m/s)',1,10,0.25,5],['Kinematic viscosity (mm²/s)',8,24,1,15],['Plate station (m)',0.05,0.4,0.01,0.2]],
    xLabel:'Wall-normal distance (mm)',yLabel:'Velocity / outer velocity',
    calculate([u,nu,station]) {
      const delta=Math.sqrt(30*nu*1e-6*station/u)*1000;
      const profile=(y,edge)=>{const t=Math.min(1,y/edge);return 2*t-t*t;};
      const cf=4/Math.sqrt(30*u*station/(nu*1e-6));
      return {curves:[curve('Selected station',0,1.6*delta,y=>profile(y,delta)),curve('Twice the distance downstream',0,1.6*delta,y=>profile(y,Math.SQRT2*delta))],
        result:'The quadratic trial layer has thickness '+delta.toFixed(2)+' mm, displacement thickness '+(delta/3).toFixed(2)+' mm and local friction coefficient '+cf.toFixed(5)+'. Smooth, laminar, zero-pressure-gradient conditions are assumed.'};
    }
  };
  studies['engineering-methods-mathieu'] = {
    controls:[['Natural frequency (Hz)',3,20,0.5,10],['Stiffness modulation fraction',0.02,0.28,0.01,0.16],['Damping ratio',0.005,0.055,0.005,0.025]],
    xLabel:'Modulation frequency / twice natural frequency',yLabel:'Amplitude growth exponent (1/s)',
    calculate([frequency,modulation,damping]) {
      const omega=2*pi*frequency;
      const rate=(ratio,h)=>omega*(-damping+Math.sqrt(Math.max(0,(h/4)**2-(ratio-1)**2)));
      const extent=Math.sqrt(Math.max(0,(modulation/4)**2-damping**2));
      return {curves:[curve('Selected modulation',0.91,1.09,r=>rate(r,modulation),360),curve('Half modulation',0.91,1.09,r=>rate(r,modulation/2),360)],
        result:'The leading-order exact-tuning exponent is '+rate(1,modulation).toFixed(3)+' per second. '+(extent>0?'The predicted unstable band is '+(2*frequency*(1-extent)).toFixed(2)+' to '+(2*frequency*(1+extent)).toFixed(2)+' Hz.':'No finite principal instability band is predicted at this modulation.')};
    }
  };
  studies['engineering-methods-random-response'] = {
    controls:[['Natural angular frequency (rad/s)',60,150,5,100],['Damping ratio',0.02,0.15,0.005,0.05],['One-sided acceleration PSD ((m/s²)²/Hz)',0.01,0.1,0.005,0.04]],
    xLabel:'Upper bandwidth (Hz)',yLabel:'RMS relative displacement (mm)',
    calculate([omega,zeta,density]) {
      const maxF=4*omega/(2*pi), n=2400,df=maxF/n;
      const integrand=f=>density/((omega**2-(2*pi*f)**2)**2+(2*zeta*omega*2*pi*f)**2);
      let integral=0;
      const points=[[0,0]];
      for(let i=1;i<=n;i++){
        integral+=df*(integrand((i-1)*df)+integrand(i*df))/2;
        if(i%6===0) points.push([i*df,1000*Math.sqrt(integral)]);
      }
      const limiting=1000*Math.sqrt(density/(8*zeta*omega**3));
      return {curves:[{name:'Integrated measured passband',points},curve('Ideal infinite white passband',0,maxF,()=>limiting)],
        result:'The ideal white-input limit is '+limiting.toFixed(3)+' mm rms; stopping the integration at '+maxF.toFixed(1)+' Hz gives '+(1000*Math.sqrt(integral)).toFixed(3)+' mm.'};
    }
  };
  studies['engineering-methods-modal-dashpot'] = {
    controls:[['Local dashpot (N s/m)',1,8,0.25,4],['Second modal frequency (rad/s)',11,22,0.5,14]],
    xLabel:'Forcing angular frequency (rad/s)',yLabel:'Modal ratio |Q₂/Q₁|',
    calculate([dashpot,w2]) {
      const ratio=w=>{const damping=dashpot*w/2;return damping/Math.hypot(w2*w2-w*w,damping);};
      return {curves:[curve('Physical local dashpot',3,1.6*w2,ratio,360),curve('Diagonal modal approximation',3,1.6*w2,()=>0)],
        result:'At the first natural frequency of 10 rad/s, the second-to-first modal amplitude ratio is '+ratio(10).toFixed(3)+'. At the undamped second frequency the ratio is '+ratio(w2).toFixed(2)+'.'};
    }
  };
  studies['engineering-methods-gear-reaction'] = {
    controls:[['Shaft torque (N m)',40,180,5,100],['Pitch diameter (mm)',60,160,5,100],['Helix angle (°)',5,35,1,20]],
    xLabel:'Mesh station along 150 mm span (mm)',yLabel:'Signed radial support force (N)',
    calculate([torque,diameter,beta]) {
      const ft=2*torque/(diameter/1000),fr=ft*Math.tan(20*pi/180)/Math.cos(beta*pi/180),fa=ft*Math.tan(beta*pi/180);
      const couple=fa*diameter/2000;
      const reactionB=x=>(fr*x/1000+couple)/0.15;
      return {curves:[curve('Support A, thrust couple included',0,150,x=>fr-reactionB(x)),curve('Support B, thrust couple included',0,150,reactionB),curve('Support B, centreline force alone',0,150,x=>fr*x/150)],
        result:'For the marked 60 mm station, radial reactions are '+(fr-reactionB(60)).toFixed(1)+' and '+reactionB(60).toFixed(1)+' N. The '+fa.toFixed(1)+' N thrust acts at pitch radius and supplies a '+couple.toFixed(2)+' N m couple.'};
    }
  };
  studies['engineering-methods-sliding-wear'] = {
    controls:[['First interval load (N)',50,250,10,200],['Second interval load (N)',20,180,10,50],['Hardness (GPa)',1,4,0.1,2]],
    xLabel:'Elapsed time (min)',yLabel:'Cumulative wear volume (mm³)',
    calculate([first,second,hardness]) {
      const coefficient=2e-5/hardness;
      const actual=t=>coefficient*60*(first*0.2*Math.min(t,30)+second*0.8*Math.max(0,t-30));
      const averaged=t=>coefficient*60*t*(first+second)/2*0.5;
      return {curves:[curve('Two successive operating conditions',0,60,actual),curve('Separate average load and average speed',0,60,averaged)],
        result:'After one hour the integrated history predicts '+actual(60).toFixed(3)+' mm³. Multiplying separate averages predicts '+averaged(60).toFixed(3)+' mm³ ('+(100*(averaged(60)/actual(60)-1)).toFixed(1)+'% difference).'};
    }
  };
  studies['engineering-methods-crack-growth'] = {
    controls:[['Stress range (MPa)',55,105,1,80],['Initial crack length (mm)',0.6,1.5,0.05,1],['Final crack length (mm)',2,5,0.1,4]],
    xLabel:'Crack length (mm)',yLabel:'Accumulated cycles (thousands)',
    calculate([stress,initial,final]) {
      const cycles=(a,amplitude)=>2*(1/Math.sqrt(initial/1000)-1/Math.sqrt(a/1000))/(1e-11*(1.1*amplitude*Math.sqrt(pi))**3)/1000;
      return {curves:[curve('Selected stress range',initial,final,a=>cycles(a,stress)),curve('Five percent higher stress range',initial,final,a=>cycles(a,1.05*stress))],
        result:'Growth over the selected length interval takes '+Math.round(cycles(final,stress)*1000).toLocaleString('en-US')+' cycles at the fitted intermediate-growth law. The endpoint growth rate increases by a factor ' + ((final/initial)**1.5).toFixed(2)+'.' };
    }
  };
  studies['engineering-methods-rga'] = {
    controls:[['Cross gain k₁₂',0.1,1.4,0.05,0.9],['Cross gain k₂₁',0.1,1.4,0.05,1.1],['Direct gain k₂₂',0.6,1.5,0.01,1]],
    xLabel:'Perturbation of input one',yLabel:'First output change',
    calculate([k12,k21,k22]) {
      const g=1-k12*k21/k22;
      return {curves:[curve('Other input fixed',-1,1,x=>x),curve('Other output fixed',-1,1,x=>g*x)],
        result:'The constrained gain is '+g.toFixed(4)+'. '+(Math.abs(g)<1e-10?'The matrix is singular; this relative gain is undefined.':'The diagonal relative gain is '+(1/g).toFixed(2)+'.')+' For a unit first input the second actuator must move '+(-k21/k22).toFixed(3)+' units to keep its output fixed.'};
    }
  };
  studies['engineering-methods-smith-delay'] = {
    controls:[['Actual transport delay (s)',1,5,0.1,3],['Model transport delay (s)',1,5,0.1,3],['Proportional gain',0.5,3,0.1,2]],
    xLabel:'Time after reference step (s)',yLabel:'Measured or predicted response',
    calculate([actual,model,gain]) {
      const dt=0.01,steps=3000,lag=Math.round(actual/dt),modelLag=Math.round(model/dt),states=[0], decayed=Math.exp(-dt/4);
      const measured=[],prediction=[],nominal=[];
      for(let i=0;i<=steps;i++){
        const t=i*dt,current=states[i],y=i>=lag?states[i-lag]:0,expected=i>=modelLag?states[i-modelLag]:0;
        const yp=y+current-expected,u=gain*(1-yp);
        states.push(current*decayed+(1-decayed)*u);
        if(i%8===0){
          measured.push([t,y]);prediction.push([t,yp]);
          nominal.push([t,t<actual?0:gain/(1+gain)*(1-Math.exp(-(1+gain)*(t-actual)/4))]);
        }
      }
      let largest=0;
      measured.forEach(([t,y],i)=>{largest=Math.max(largest,Math.abs(y-nominal[i][1]));});
      return {curves:[{name:'Physical sensor',points:measured},{name:'Predictor feedback signal',points:prediction},{name:'Exact-model reference response',points:nominal}],
        result:'The largest measured deviation from an exact-delay benchmark during 30 s is '+largest.toFixed(3)+'. The driver updates every 0.01 s. No physical response is possible before '+actual.toFixed(1)+' s.'};
    }
  };
  studies['engineering-methods-sensitivity'] = {
    controls:[['Loop numerator K',0.5,12,0.25,4],['First pole frequency (rad/s)',0.5,2.5,0.1,1],['Second pole frequency (rad/s)',1,4,0.1,2]],
    xLabel:'Angular frequency (rad/s)',yLabel:'Frequency-response magnitude',
    calculate([gain,a,b]) {
      const den=w=>Math.hypot(a*b+gain-w*w,(a+b)*w);
      const sensitivity=w=>Math.hypot(a*b-w*w,(a+b)*w)/den(w);
      const complementary=w=>gain/den(w);
      const limit=5*Math.sqrt(a*b+gain);
      const S=curve('Output disturbance |S|',0,limit,sensitivity,600);
      const peak=S.points.reduce((best,p)=>p[1]>best[1]?p:best);
      return {curves:[S,curve('Reference or sensor noise |T|',0,limit,complementary,600),curve('Unit magnitude',0,limit,()=>1)],
        result:'Static sensitivity is '+sensitivity(0).toFixed(3)+'. The sampled peak is '+peak[1].toFixed(3)+' near '+peak[0].toFixed(2)+' rad/s. At 3 rad/s |S| = '+sensitivity(3).toFixed(3)+'.'};
    }
  };
})();
