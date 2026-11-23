(() => {
  'use strict';
  const studies = window.fundamentalStudies ||= {};
  const pi = Math.PI;
  const radians = degrees => degrees * pi / 180;
  const fmt = (value, digits = 2) => (Math.abs(value) < 0.5 * 10 ** -digits ? 0 : value).toFixed(digits);

  function curve(name, start, end, valueAt, intervals = 100) {
    const points = Array.from({length: intervals + 1}, (_, i) => {
      const x = start + (end - start) * i / intervals;
      return [x, valueAt(x)];
    });
    return {name, points};
  }

  function register(course, slug, controls, xLabel, yLabel, calculate) {
    studies['eng-' + course + '-' + slug] = {controls, xLabel, yLabel, calculate};
  }

  const mechanics = 'engineering-mechanics';
  register(mechanics, 'local-equilibrium',
    [['Normal stress xx (MPa)', 10, 100, 5, 60], ['Shear stress xy (MPa)', -40, 40, 2, 10]],
    'Cut normal angle (°)', 'Traction (MPa)', ([xx, xy]) => {
      const yy = 20;
      const normal = angle => (xx + yy) / 2 + (xx - yy) / 2 * Math.cos(2*radians(angle)) + xy*Math.sin(2*radians(angle));
      const tangent = angle => xy*Math.cos(2*radians(angle)) - (xx - yy) / 2*Math.sin(2*radians(angle));
      const angle = Math.atan2(0.8, 0.6) * 180 / pi;
      return {curves: [curve('Normal traction',0,180,normal),curve('Signed shear',0,180,tangent)],
        result: 'On the cut with normal (0.6, 0.8), the normal traction is ' + fmt(normal(angle)) + ' MPa and the signed shear is ' + fmt(tangent(angle)) + ' MPa.'};
    });

  register(mechanics, 'elastic-constraints',
    [['Young’s modulus (MPa)', 10, 80, 2, 30], ['Poisson ratio', 0, 0.48, 0.01, 0.4]],
    'Applied compression (MPa)', 'Contraction (mm)', ([E, nu]) => {
      const M = E*(1-nu)/((1+nu)*(1-2*nu));
      return {curves: [curve('Free sides',0,0.5,p=>20*p/E), curve('Rigid sleeve',0,0.5,p=>20*p/M)],
        result: 'At 0.30 MPa the 20 mm specimen contracts by ' + fmt(6/E,3) + ' mm freely and ' + fmt(6/M,3) + ' mm in the sleeve. The sleeve modulus is ' + fmt(M) + ' MPa.'};
    });

  register(mechanics, 'annular-elasticity',
    [['Outer radius (mm)', 23, 65, 1, 40], ['Inner pressure (MPa)', 1, 8, 0.25, 3]],
    'Radius (mm)', 'Stress (MPa)', ([b, p]) => {
      const a = 20, A = p*a*a/(b*b-a*a), B = A*b*b;
      return {curves: [curve('Hoop stress',a,b,r=>A+B/r**2), curve('Radial stress',a,b,r=>A-B/r**2),
        curve('Mean hoop stress',a,b,()=>p*a/(b-a))],
        result: 'The inside hoop stress is ' + fmt(A+B/a**2) + ' MPa. The wall average is ' + fmt(p*a/(b-a)) + ' MPa, while the radial stress vanishes at the outer boundary.'};
    });

  register(mechanics, 'viscoelastic-response',
    [['Viscosity (MPa s)', 600, 4800, 100, 2400], ['Parallel spring modulus (MPa)', 10, 60, 2, 30]],
    'Holding time (s)', 'Fraction of initial transient', ([viscosity,E0]) => {
      const relax = viscosity/60, creep = relax*(E0+60)/E0;
      return {curves: [curve('Relaxation excess stress',0,300,t=>Math.exp(-t/relax)),curve('Creep strain still to develop',0,300,t=>Math.exp(-t/creep))],
        result: 'The relaxation time is ' + fmt(relax,1) + ' s and the creep time is ' + fmt(creep,1) + ' s. At 1% fixed strain the stress after 80 s is ' + fmt(0.01*(E0+60*Math.exp(-80/relax)),3) + ' MPa.'};
    });

  register(mechanics, 'virtual-work-elements',
    [['Number of elements',1,16,1,2],['Distributed load (N/m)',200,2000,100,1200]], 'Bar position (m)', 'Axial force (N)', ([n,q]) => {
      const pairs = [];
      for (let i=0;i<n;i++) {
        const force=q*(1-(i+0.5)/n);
        pairs.push([i/n,force],[(i+1)/n,force]);
      }
      return {curves: [curve('Exact equilibrium',0,1,x=>q*(1-x)),{name:'Element force',points:pairs}],
        result: 'The tip displacement is ' + fmt(1000*q/(2*2e6),3) + ' mm on every mesh here. Maximum axial-force error is ' + fmt(q/(2*n),1) + ' N with ' + n + ' elements.'};
    });

  const fluids = 'fluid-mechanics';
  register(fluids, 'oscillating-wall',
    [['Wall frequency (Hz)',0.5,6,0.25,2],['Cycle phase (°)',0,360,5,0]], 'Distance above wall (mm)', 'Fluid velocity (m/s)', ([f,angle]) => {
      const delta=1000*Math.sqrt(5e-5/(pi*f)), envelope=y=>0.08*Math.exp(-y/delta);
      return {curves:[curve('Velocity snapshot',0,12,y=>envelope(y)*Math.cos(radians(angle)-y/delta)),
        curve('Positive envelope',0,12,envelope),curve('Negative envelope',0,12,y=>-envelope(y))],
        result:'The penetration depth is ' + fmt(delta) + ' mm. The 3 mm probe has a speed amplitude of ' + fmt(envelope(3),4) + ' m/s and lags by ' + fmt(3/delta/(2*pi*f),3) + ' s.'};
    });

  register(fluids, 'potential-flow',
    [['Incident speed (m/s)',0.2,1.4,0.05,0.8],['Rear pressure deficit (Pa)',0,400,20,0]],
    'Surface normal angle from downstream (°)', 'Gauge pressure (Pa)', ([U,deficit]) => {
      const dynamic=500*U**2, ideal=angle=>dynamic*(1-4*Math.sin(radians(angle))**2);
      const modified=angle=>ideal(angle)-(Math.cos(radians(angle))>0?deficit:0);
      return {curves:[curve('Potential-flow pressure',0,360,ideal,180),curve('With imposed rear deficit',0,360,modified,180)],
        result:'The ideal front-to-side difference is '+fmt(4*dynamic,0)+' Pa. Lowering pressure on the downstream semicircle gives a pressure-drag coefficient '+fmt(deficit/dynamic,3)+' based on diameter. The added deficit is an imposed measurement model.'};
    });

  register(fluids, 'surface-tension',
    [['Contact angle (°)',0,145,1,30],['Surface tension (mN/m)',30,80,1,72]],
    'Tube radius (mm)', 'Equilibrium rise (mm)', ([angle,sigma]) => {
      const rise=(r,theta)=>2000*(sigma/1000)*Math.cos(radians(theta))/(1000*9.81*r/1000);
      return {curves:[curve('Selected wetting angle',0.2,1.5,r=>rise(r,angle)),curve('Complete wetting',0.2,1.5,r=>rise(r,0))],
        result:'At a radius of 0.400 mm the level change is '+fmt(rise(0.4,angle),1)+' mm relative to the reservoir. Positive values indicate rise; negative values indicate depression.'};
    });

  register(fluids, 'free-surface-control',
    [['Discharge per width (m²/s)',0.1,0.7,0.025,0.4],['Bed rise (m)',0,0.25,0.005,0.1]],
    'Water depth (m)', 'Specific energy (m)', ([q,bump]) => {
      const hc=Math.cbrt(q*q/9.81),energy=h=>h+q*q/(2*9.81*h*h), available=energy(0.5)-bump;
      return {curves:[curve('Required energy',0.4*hc,Math.max(0.8,2.5*hc),energy),curve('Head over bump',0.4*hc,Math.max(0.8,2.5*hc),()=>available)],
        result:'Critical depth is '+fmt(hc,3)+' m. The available head over the bump is '+fmt(available,3)+' m versus a minimum of '+fmt(1.5*hc,3)+' m. '+(available>=1.5*hc?'The specified upstream state can pass this discharge.':'The specified upstream state must adjust.')};
    });

  register(fluids, 'lubrication-films',
    [['Disk gap (mm)',0.1,0.4,0.01,0.2],['Approach speed (mm/s)',0.025,0.3,0.025,0.1]],
    'Distance from disk centre (mm)', 'Gauge pressure (kPa)', ([gap,speed]) => {
      const radius=0.02,center=3*0.1*(speed/1000)*radius**2/(gap/1000)**3;
      return {curves:[curve('Film pressure',0,20,r=>center*(1-(r/20)**2)/1000),curve('Area-average pressure',0,20,()=>center/2000)],
        result:'The central excess pressure is '+fmt(center/1000,3)+' kPa and the integrated disk load is '+fmt(pi*radius**2*center/2,3)+' N.'};
    });

  const thermo = 'engineering-thermodynamics';
  register(thermo, 'gas-cycle',
    [['Compressor efficiency',0.65,1,0.01,0.8],['Turbine efficiency',0.75,1,0.01,0.9]],'Pressure ratio','Net shaft work (kJ/kg)',([ec,et])=>{
      const work=(r,c,t)=>{const x=r**(2/7); return 1.005*(1200*t*(1-1/x)-300*(x-1)/c);};
      const optimum=(4*ec*et)**1.75;
      return {curves:[curve('Ideal',1,32,r=>work(r,1,1)),curve('Selected efficiencies',1,32,r=>work(r,ec,et))],
        result:'The best pressure ratio for net work with these component efficiencies is '+fmt(optimum)+', giving '+fmt(work(optimum,ec,et),1)+' kJ/kg. At the ideal optimum 11.31 the real cycle gives '+fmt(work(2**3.5,ec,et),1)+' kJ/kg.'};
    });

  register(thermo, 'throttling',
    [['Attractive constant a (Pa m⁶/mol²)',0.04,0.12,0.005,0.08],['Pressure drop (bar)',0.5,3,0.25,2]],
    'Inlet temperature (K)','Outlet minus inlet temperature (K)',([a,drop])=>{
      const change=T=>-drop*1e5*(2*a/(8.314*T)-4e-5)/30;
      return {curves:[curve('Low-density real-gas estimate',230,750,change),curve('Ideal gas',230,750,()=>0)],
        result:'The inversion temperature is '+fmt(2*a/(8.314*4e-5),1)+' K. At 300 K the local pressure-drop estimate gives an outlet temperature change of '+fmt(change(300),3)+' K.'};
    });

  register(thermo, 'fins',
    [['Length (mm)',20,200,5,50],['Convection coefficient (W/m² K)',5,75,2.5,25]],
    'Distance from root (mm)','Temperature above ambient (K)',([L,h])=>{
      const m=Math.sqrt(h*0.044/(200*4e-5)), mL=m*L/1000, profile=x=>50*Math.cosh(m*(L-x)/1000)/Math.cosh(mL);
      return {curves:[curve('Insulated-tip temperature',0,L,profile),curve('Infinite-fin approximation',0,L,x=>50*Math.exp(-m*x/1000))],
        result:'The finite tip is '+fmt(profile(L),1)+' K above ambient. Root heat flow is '+fmt(200*4e-5*m*50*Math.tanh(mL))+' W and the fin efficiency is '+fmt(100*Math.tanh(mL)/mL,1)+'%.'};
    });

  register(thermo, 'freezing-front',
    [['Coolant depression below freezing (K)',1,10,0.5,5],['Film coefficient (W/m² K)',50,600,10,200]],
    'Time (min)','Ice thickness (mm)',([delta,h])=>{
      const k=2.2,rhoL=917*334000, filmLength=k/h;
      const ideal=t=>Math.sqrt(2*k*delta*t*60/rhoL);
      const finite=t=> {const a=2*k*delta*t*60/rhoL; return a/(Math.sqrt(filmLength**2+a)+filmLength);};
      const t10=rhoL/delta*(0.01/h+0.01**2/(2*k))/60;
      return {curves:[curve('Specified wall temperature',0,150,t=>1000*ideal(t)),curve('Finite coolant film',0,150,t=>1000*finite(t))],
        result:'At the selected film coefficient 10 mm requires '+fmt(t10,1)+' minutes. The fixed-wall prediction is '+fmt(rhoL*0.01**2/(2*k*delta)/60,1)+' minutes.'};
    });

  register(thermo, 'stagnant-film',
    [['Surface vapour mole fraction',0.1,0.85,0.05,0.5],['Reservoir vapour mole fraction',0,0.09,0.01,0]],
    'Distance from liquid (mm)','Vapour mole fraction',([surface,far])=>{
      const ratio=(1-far)/(1-surface), coefficient=1e5/(8.314*300)*2e-5/0.005;
      return {curves:[curve('Stationary carrier',0,5,x=>1-(1-surface)*ratio**(x/5)),curve('Linear interpolation',0,5,x=>surface+(far-surface)*x/5)],
        result:'The evaporation flux is '+fmt(coefficient*Math.log(ratio),4)+' mol/(m² s). The dilute estimate is '+fmt(coefficient*(surface-far),4)+' mol/(m² s).'};
    });

  const dynamics = 'machine-dynamics';
  register(dynamics, 'elastic-waves',
    [['Second / first section area',0.25,4,0.05,2],['Incident peak velocity (m/s)',0.1,0.6,0.025,0.3]],
    'Time at interface (ms)','Particle velocity (m/s)',([ratio,peak])=>{
      const reflection=(1-ratio)/(1+ratio),transmission=1+reflection;
      const pulse=t=>peak*Math.exp(-((t-0.5)**2)/(2*0.12**2));
      return {curves:[curve('Incident component',0,1,pulse),curve('Reflected component',0,1,t=>reflection*pulse(t)),curve('Transmitted component',0,1,t=>transmission*pulse(t))],
        result:'Peak reflected velocity is '+fmt(reflection*peak,3)+' m/s and transmission is '+fmt(transmission*peak,3)+' m/s. The reflection carries '+fmt(100*reflection**2,1)+'% of the incident energy.'};
    });

  register(dynamics, 'bending-waves',
    [['Span (m)',0.6,1.4,0.05,1], ['Thickness (mm)',2,8,0.25,4]],
    'Position along beam (m)','Normalised modal displacement',([L,h])=>{
      const fundamental=pi/(2*L**2)*(h/1000)*Math.sqrt(200e9/(12*7800));
      return {curves:[1,2,3].map(n=>curve('Mode '+n+' · '+fmt(n**2*fundamental,1)+' Hz',0,L,x=>Math.sin(n*pi*x/L))),
        result:'At midspan the second mode has a displacement node. The three frequencies are '+[1,4,9].map(f=>fmt(f*fundamental,2)).join(', ')+' Hz.'};
    });

  register(dynamics, 'rayleigh-estimate',
    [['Attached mass (kg)',0.01,0.12,0.005,0.05],['Bending rigidity (N m²)',5,20,0.5,10]],
    'Mass position / span','Estimated frequency (Hz)',([M,EI])=>{
      const phi=s=>s*s*(3-s)/2, k=3*EI/0.5**3, distributed=0.4*33/140;
      const f=s=>Math.sqrt(k/(distributed+M*phi(s)**2))/(2*pi);
      return {curves:[curve('Attached mass',0,1,f),curve('Bare beam estimate',0,1,()=>f(0))],
        result:'The estimate is '+fmt(f(1))+' Hz with the sensor at the tip and '+fmt(f(0.5))+' Hz at half-span. The bare-beam estimate is '+fmt(f(0))+' Hz.'};
    });

  register(dynamics, 'moving-frame',
    [['Angular speed (rad/s)',1,10,0.5,5],['Outward speed (m/s)',0,0.4,0.025,0.2]],
    'Time from start (s)','Force on carriage (N)',([omega,u])=>{
      const radial=t=>-0.4*(0.10+u*t)*omega**2, side=2*0.4*u*omega;
      return {curves:[curve('Radial drive force',0,1,radial),curve('Tangential guide force',0,1,()=>side)],
        result:'At 0.50 s the radial and tangential forces are '+fmt(radial(0.5))+' N and '+fmt(side)+' N. The net mechanical power is '+fmt(0.4*(0.1+u*0.5)*u*omega**2)+' W.'};
    });

  register(dynamics, 'gyroscopic-motion',
    [['Tilt from vertical (°)',25,80,1,60],['Moment arm (mm)',20,100,5,60]],
    'Spin component (rad/s)','Slow precession (rad/s)',([angle,arm])=>{
      const mgl=0.5*9.81*arm/1000, I3=0.002, term=4*0.003*mgl*Math.cos(radians(angle)), threshold=Math.sqrt(term)/I3;
      const slow=s=>2*mgl/(I3*s+Math.sqrt(Math.max(0,(I3*s)**2-term)));
      return {curves:[curve('Exact steady slow branch',threshold,120,slow,200),curve('Fast-spin approximation',threshold,120,s=>mgl/(I3*s),200)],
        result:'For this tilt the two regular-precession branches first exist at spin '+fmt(threshold,1)+' rad/s. At 100 rad/s the exact slow rate is '+fmt(slow(100),3)+' rad/s.'};
    });

  const elements = 'machine-elements';
  register(elements, 'kinematic-location',
    [['Height error at pad 2 (μm)',-25,25,1,12],['Pad spacing (mm)',60,180,5,120]],
    'Feature position along x (mm)','Vertical shift (μm)',([height,a])=>{
      return {curves:[curve('Tilted datum',0,200,x=>height*x/a),curve('Nominal datum',0,200,()=>0)],
        result:'Pitch is '+fmt(-1000*height/a,1)+' μrad. At 180 mm the feature shifts by '+fmt(180*height/a,2)+' μm; at the second support it shifts by '+fmt(height,1)+' μm.'};
    });

  register(elements, 'parallel-flexures',
    [['Leaf thickness (mm)',0.15,0.65,0.025,0.3],['Stage stroke (mm)',0.2,1.5,0.05,1]],
    'Position along leaf (mm)','Outer-fibre normal stress (MPa)',([thickness,stroke])=>{
      const t=thickness/1000,d=stroke/1000,L=0.04,E=200e9, stress=3*E*t*d/L**2/1e6;
      const force=24*E*(0.01*t**3/12)*d/L**3;
      return {curves:[curve('One outer face',0,40,x=>stress*(1-2*x/40)),curve('Opposite face',0,40,x=>-stress*(1-2*x/40))],
        result:'The guide force is '+fmt(force)+' N, peak nominal stress is '+fmt(stress,1)+' MPa and projected shortening is '+fmt(1e6*3*d*d/(5*L),1)+' μm.'};
    });

  register(elements, 'screw-drives',
    [['Sliding friction coefficient',0.04,0.22,0.01,0.12],['Axial load (N)',200,1600,50,800]],
    'Thread lead (mm)','Applied torque (N m)',([mu,W])=>{
      const t=l=>l/(pi*16),raising=l=>W*0.008*(mu+t(l))/(1-mu*t(l)),lowering=l=>W*0.008*(mu-t(l))/(1+mu*t(l));
      return {curves:[curve('Raise',1,14,raising),curve('Lower (positive assists descent)',1,14,lowering)],
        result:'At 4 mm lead, raising takes '+fmt(raising(4),3)+' N m and lowering takes '+fmt(lowering(4),3)+' N m. The ideal change of lowering-torque sign is at '+fmt(pi*16*mu)+' mm lead.'};
    });

  register(elements, 'motor-matching',
    [['Resisting torque (N m)',0,1.5,0.1,0],['Load inertia (kg m²)',0.01,0.06,0.002,0.02]],
    'Reduction ratio','Load acceleration (rad/s²)',([load,JL])=>{
      const Jm=0.0002, torque=0.4;
      const acceleration=(N,d)=>(N*torque-d)/(JL+Jm*N**2);
      const optimum=load/torque+Math.sqrt((load/torque)**2+JL/Jm);
      return {curves:[curve('With selected load torque',1,35,n=>acceleration(n,load)),curve('Inertia-only benchmark',1,35,n=>acceleration(n,0))],
        result:'For constant motor torque the maximum acceleration is '+fmt(acceleration(optimum,load),1)+' rad/s² at a ratio of '+fmt(optimum)+'.'};
    });

  register(elements, 'toggle-linkage',
    [['Input force (N)',20,100,5,60],['Spring stiffness (N/mm)',50,200,5,100]],
    'Link angle (°)','Horizontal reaction (N)',([P,stiffness])=>{
      const ideal=a=>P/(2*Math.tan(radians(a)));
      const elastic=a=>stiffness*Math.max(0,200*(Math.cos(radians(a))-Math.cos(radians(10))));
      return {curves:[curve('Given input force',2,18,ideal),curve('Given spring displacement',2,18,elastic)],
        result:'At 5° a specified '+fmt(P,0)+' N input would balance '+fmt(ideal(5),1)+' N. The spring first touching at 10° instead carries '+fmt(elastic(5),1)+' N and requires '+fmt(2*elastic(5)*Math.tan(radians(5)),1)+' N input.'};
    });

  const control = 'control-engineering';
  register(control, 'energy-stability',
    [['Initial angular speed (rad/s)',0,8,0.1,4],['Total viscous damping (N m s/rad)',0,0.8,0.025,0.3]],
    'Time (s)','Mechanical energy (J)',([initial,damping])=>{
      const J=0.2,K=2,dt=0.005,energy=(q,v)=>J*v*v/2+K*(1-Math.cos(q));
      let q=pi/3,v=initial;
      const points=[[0,energy(q,v)]],a=(angle,speed)=>(-K*Math.sin(angle)-damping*speed)/J;
      for (let i=1;i<=1000;i++) {
        const q1=v,v1=a(q,v),q2=v+dt*v1/2,v2=a(q+dt*q1/2,v+dt*v1/2);
        const q3=v+dt*v2/2,v3=a(q+dt*q2/2,v+dt*v2/2),q4=v+dt*v3,v4=a(q+dt*q3,v+dt*v3);
        q+=dt*(q1+2*q2+2*q3+q4)/6;
        v+=dt*(v1+2*v2+2*v3+v4)/6;
        if(i%5===0) points.push([i*dt,energy(q,v)]);
      }
      return {curves:[{name:'Integrated nonlinear pendulum',points},curve('Upright energy barrier',0,5,()=>4)],
        result:'The 60° release starts at '+fmt(points[0][1],3)+' J and has '+fmt(energy(q,v),3)+' J after 5 s. '+(points[0][1]<4?'Its starting sublevel stays inside the initial well.':'Its initial energy exceeds or reaches the sufficient bound.')};
    });

  register(control, 'integral-load',
    [['Critical pole rate (1/s)',2,10,0.5,4],['Opposing load step (N m)',0.5,3,0.25,2]],
    'Time (s)','Speed deviation (rad/s)',([a,d])=>{
      const J=0.2,totalDrag=2*J*a;
      return {curves:[curve('PI load response',0,2,t=>-d/J*t*Math.exp(-a*t)),curve('Same proportional gain alone',0,2,t=>-d/totalDrag*(1-Math.exp(-totalDrag*t/J)))],
        result:'The PI deviation reaches '+fmt(-d/(J*Math.E*a),3)+' rad/s after '+fmt(1/a,3)+' s and tends to zero. The proportional-only offset is '+fmt(-d/totalDrag,3)+' rad/s.'};
    });

  register(control, 'reference-shaping',
    [['Estimated inertia (kg m²)',0.05,0.35,0.01,0.15],['Move duration (s)',0.5,2,0.1,1]],
    'Time (s)','Speed (rad/s)',([estimate,T])=>{
      const J=0.20, b=0.5, kp=1.5, end=T+0.5, steps=600, dt=end/steps;
      const ref=t=>{const s=Math.min(1,Math.max(0,t/T));return [4*(3*s*s-2*s**3),24*s*(1-s)/T];};
      const feedback=[], feedforward=[], target=[];
      let eFeed=0,eP=0;
      const decay=Math.exp(-(b+kp)*dt/J);
      for(let i=0;i<=steps;i++) {
        const t=i*dt,[r]=ref(t);
        if(i%4===0){target.push([t,r]);feedback.push([t,r-eP]);feedforward.push([t,r-eFeed]);}
        const [mid,rate]=ref(t+dt/2);
        // Integrate the scalar error equation with frozen midpoint forcing.
        eFeed=eFeed*decay+(1-decay)*(J-estimate)*rate/(b+kp);
        eP=eP*decay+(1-decay)*(J*rate+b*mid)/(b+kp);
      }
      return {curves:[{name:'Cubic reference',points:target},{name:'Feedback + feedforward',points:feedforward},{name:'Proportional feedback alone',points:feedback}],
        result:'The midpoint inertial-plus-drag feedforward command is '+fmt(estimate*6/T+1)+' N m. The completed hold requires 2.00 N m at the target speed.'};
    });

  register(control, 'delay-margin',
    [['Loop gain',1.25,5,0.25,2],['Delay (s)',0,0.8,0.025,0.3]],
    'Frequency (rad/s)','Unwrapped loop phase (°)',([K,delay])=>{
      const T=0.5,wc=Math.sqrt(K*K-1)/T,phase=(w,t)=>(-Math.atan(w*T)-w*t)*180/pi;
      const crit=(pi-Math.atan(wc*T))/wc;
      return {curves:[curve('Loop with delay',0,2*wc,w=>phase(w,delay)),curve('Undelayed loop',0,2*wc,w=>phase(w,0)),curve('Critical phase',0,2*wc,()=>-180)],
        result:'Gain crossover is '+fmt(wc,3)+' rad/s. The remaining phase margin is '+fmt(180+phase(wc,delay),1)+'° and the first critical total delay is '+fmt(crit,3)+' s.'};
    });

  register(control, 'inverse-response',
    [['Fast-path time constant (s)',0.25,3,0.05,0.5],['Subtracted gain',0.5,1.5,0.05,1]],
    'Time (s)','Unit-step response',([Tf,b])=>{
      const slow=t=>2*(1-Math.exp(-t/4)),fast=t=>b*(1-Math.exp(-t/Tf)),output=t=>slow(t)-fast(t);
      const initial=0.5-b/Tf, min=initial<0?Math.log(b*4/(2*Tf))/(1/Tf-0.25):0;
      return {curves:[curve('Measured difference',0,12,output,200),curve('Slow path',0,12,slow,200),curve('Subtracted fast path',0,12,t=>-fast(t),200)],
        result:'Initial slope is '+fmt(initial,3)+' per second and final gain is '+fmt(2-b,2)+'. '+(initial<0?'The minimum is '+fmt(output(min),3)+' at '+fmt(min,3)+' s.':'The response rises without a negative initial excursion.')};
    });

  register(mechanics, 'energy-reciprocity',
    [['Left bar compliance (μm/N)',1,6,0.25,2],['Right bar compliance (μm/N)',1,6,0.25,4]],
    'Downward force (N)','Joint travel (mm)',([left,right])=>{
      const direct=(left+right)/2000,cross=(left-right)/2000;
      return {curves:[curve('Downward travel under vertical force',0,200,v=>direct*v),curve('Rightward travel under vertical force',0,200,v=>cross*v)],
        result:'At 100 N down the joint travels '+fmt(100*direct,3)+' mm down and '+fmt(100*cross,3)+' mm right. By reciprocity the latter also gives downward travel under 100 N right.'};
    });

  register(mechanics, 'thermal-curvature',
    [['Temperature change (K)',-40,80,5,40],['Upper expansion coefficient (μm/m/K)',10,26,1,20]],
    'Thickness coordinate (mm)','Axial stress (MPa)',([delta,alpha])=>{
      const mean=(alpha+10)*1e-6*delta/2,curvature=3*(alpha-10)*1e-6*delta/(4*0.0005);
      const stress=(z,a)=>1e5*(mean+curvature*z/1000-a*1e-6*delta);
      return {curves:[curve('Lower layer',-0.5,0,z=>stress(z,10)),curve('Upper layer',0,0.5,z=>stress(z,alpha))],
        result:'Free curvature is '+fmt(curvature,3)+' m⁻¹. With a 100 mm reference tangent, the small-slope tip offset is '+fmt(curvature*5,2)+' mm. The two interface stresses are '+fmt(stress(0,10),1)+' and '+fmt(stress(0,alpha),1)+' MPa.'};
    });

  register(fluids, 'wall-turbulence',
    [['Friction velocity (m/s)',0.02,0.08,0.0025,0.04],['Kinematic viscosity (mm²/s)',0.5,2,0.1,1]],
    'Wall distance (mm, two validity ranges)','Streamwise speed (m/s)',([utau,nu])=>{
      const unit=nu/1000/utau;
      const visc=y=>utau*y/unit, log=y=>utau*(Math.log(y/unit)/0.41+5);
      return {curves:[curve('Viscous range (y⁺ ≤ 5)',0,5*unit,visc),curve('Overlap estimate (30 ≤ y⁺ ≤ 180)',30*unit,180*unit,log)],
        result:'In water the wall stress is '+fmt(1000*utau**2,3)+' Pa. At 1 and 4 mm, y⁺ is '+fmt(1/unit,1)+' and '+fmt(4/unit,1)+'. Where both points lie in the overlap layer the predicted speed difference is '+fmt(utau*Math.log(4)/0.41,4)+' m/s.'};
    });

  register(fluids, 'surface-waves',
    [['Surface tension (mN/m)',35,80,1,72],['Wavelength to time (mm)',5,120,1,10]],
    'Wavelength (mm)','Deep-water wave speed (m/s)',([sigma,probe])=>{
      const cap=sigma/1e6;
      const phase=L=>{const k=2000*pi/L;return Math.sqrt(9.81/k+cap*k);};
      const group=L=>{const k=2000*pi/L;return (9.81+3*cap*k*k)/(2*k*phase(L));};
      return {curves:[curve('Phase speed',4,120,phase,200),curve('Group speed',4,120,group,200)],
        result:'For a '+fmt(probe,0)+' mm wavelength a packet crosses 0.50 m in '+fmt(0.5/group(probe),2)+' s. Phase speed is '+fmt(phase(probe),3)+' m/s. The minimum occurs at '+fmt(2000*pi*Math.sqrt(cap/9.81),2)+' mm.'};
    });

  register(thermo, 'vapour-refrigeration',
    [['Compressor isentropic efficiency',0.55,0.95,0.025,0.75],['Selected liquid enthalpy drop (kJ/kg)',0,25,1,10]],
    'Liquid enthalpy drop before valve (kJ/kg)','Refrigeration COP',([eta,drop])=>{
      const actual=d=>(150+d)*eta/30, ideal=d=>(150+d)/30;
      return {curves:[curve('Selected compressor',0,25,actual),curve('Isentropic compressor',0,25,ideal)],
        result:'At the selected '+fmt(drop,0)+' kJ/kg subcooling enthalpy drop, COP is '+fmt(actual(drop),2)+', valve-exit vapour quality is '+fmt((50-drop)/200,3)+' and a 6 kW duty takes '+fmt(6/actual(drop),3)+' kW shaft power.'};
    });

  register(thermo, 'flash-equilibrium',
    [['Feed fraction of A',0.1,0.9,0.025,0.5],['Equilibrium ratio K of A',1.25,4,0.05,2]],
    'Trial vapour fraction','Material-balance residual',([z,K])=>{
      const residual=b=>z*(K-1)/(1+b*(K-1))-(1-z)*0.5/(1-0.5*b);
      let result;
      if(residual(0)<=1e-12) result='The physical outlet is liquid (or at its bubble point).';
      else if(residual(1)>=-1e-12) result='The physical outlet is vapour (or at its dew point).';
      else {
        let lo=0,hi=1;
        for(let i=0;i<48;i++){const mid=(lo+hi)/2;if(residual(mid)>0)lo=mid;else hi=mid;}
        const beta=(lo+hi)/2,x=z/(1+beta*(K-1));
        result='Bracketed balance gives vapour fraction '+fmt(beta,4)+'. A fractions in the liquid and vapour are '+fmt(x,4)+' and '+fmt(K*x,4)+'.';
      }
      return {curves:[curve('Rachford–Rice residual',0,1,residual),curve('Zero balance',0,1,()=>0)],result};
    });

  register(dynamics, 'dry-friction',
    [['Release displacement (mm)',4,55,1,40],['Sliding friction (N)',0.2,0.8,0.05,0.5]],
    'Time (s)','Displacement (mm)',([amplitude,friction])=>{
      const k=100,m=1,staticForce=1.3*friction,omega=Math.sqrt(k/m),half=pi/omega;
      const turns=[amplitude/1000];
      while(k*Math.abs(turns[turns.length-1])>staticForce+1e-12){
        const last=turns[turns.length-1];
        turns.push(2*Math.sign(last)*friction/k-last);
      }
      const stop=(turns.length-1)*half;
      const position=t=>{
        const j=Math.min(Math.floor(t/half),turns.length-1),A=turns[j];
        if(j===turns.length-1)return 1000*A;
        const centre=Math.sign(A)*friction/k;
        return 1000*(centre+(A-centre)*Math.cos(omega*(t-j*half)));
      };
      const tmax=Math.max(0.8,stop+half);
      return {curves:[curve('Sliding segments and final hold',0,tmax,position,300),curve('Upper static limit',0,tmax,()=>1000*staticForce/k),curve('Lower static limit',0,tmax,()=>-1000*staticForce/k)],
        result:'With static friction 1.30 times sliding friction, sticking begins at '+fmt(stop,3)+' s and '+fmt(1000*turns[turns.length-1],2)+' mm. A full sliding half-cycle lasts '+fmt(half,3)+' s.'};
    });

  register(dynamics, 'periodic-loading',
    [['Pulse duty fraction',0.10,0.50,0.005,0.25],['Repetition frequency (Hz)',3,6,0.05,4]],
    'Time within one cycle (s)','Steady displacement (mm)',([d,f])=>{
      const amplitude=n=>20*Math.sin(n*pi*d)/(n*pi),zeta=0.04;
      function motion(t,count){
        let mm=10*d;
        for(let n=1;n<=count;n++){
          const r=n*f/12,real=1-r*r,imag=2*zeta*r,angle=2*pi*n*f*t;
          mm+=amplitude(n)*(real*Math.cos(angle)+imag*Math.sin(angle))/(real*real+imag*imag);
        }
        return mm;
      }
      const r=3*f/12,third=Math.abs(amplitude(3))/Math.hypot(1-r*r,2*zeta*r);
      return {curves:[curve('Mean + 24 harmonics',0,1/f,t=>motion(t,24),240),curve('Mean + fundamental',0,1/f,t=>motion(t,1),240)],
        result:'The static offset is '+fmt(10*d,2)+' mm. The third force harmonic is '+fmt(amplitude(3),3)+' N and its displacement amplitude is '+fmt(third,2)+' mm at '+fmt(3*f,2)+' Hz.'};
    });

  register(elements, 'involute-contact',
    [['Centre distance increase (mm)',0,0.8,0.025,0],['Addendum coefficient',0.8,1.1,0.025,1]],
    'Centre distance increase (mm)','Transverse contact ratio',([extension,add])=>{
      const base=Math.cos(radians(20)),pb=2*pi*base;
      const ratio=(e,c)=>([20,40].reduce((s,r)=>s+Math.sqrt((r+2*c)**2-(r*base)**2),0)-Math.sqrt((60+e)**2-(60*base)**2))/pb;
      const eps=ratio(extension,add),op=Math.acos(60*base/(60+extension))*180/pi;
      return {curves:[curve('Selected addendum',0,0.8,e=>ratio(e,add)),curve('Standard addendum',0,0.8,e=>ratio(e,1)),curve('Continuity threshold',0,0.8,()=>1)],
        result:'At the selected centre distance the operating pressure angle is '+fmt(op,2)+'° and contact ratio is '+fmt(eps,3)+'. '+(eps>=1?'The ideal two-pair fraction is '+fmt(100*(eps-1),1)+'%.':'The ideal geometry has intervals without a tooth pair in contact.')};
    });

  register(elements, 'bonded-overlaps',
    [['Overlap length (mm)',10,60,1,20],['Adhesive thickness (mm)',0.1,0.5,0.025,0.2]],
    'Position from overlap midpoint (mm)','Adhesive shear stress (MPa)',([L,t])=>{
      const lam=Math.sqrt(2*300/(t*70000*2)),mid=50*lam/(2*Math.sinh(lam*L/2)),actual=x=>mid*Math.cosh(lam*x);
      return {curves:[curve('Shear-lag field',-L/2,L/2,actual),curve('Uniform average',-L/2,L/2,()=>50/L)],
        result:'Transfer length is '+fmt(1/lam,2)+' mm. The edge stress is '+fmt(actual(L/2),3)+' MPa compared with '+fmt(mid,3)+' MPa at the centre and '+fmt(50/L,3)+' MPa averaged over the overlap.'};
    });

  register(control, 'phase-lead',
    [['Lead at crossover (°)',10,55,1,30],['Gain crossover (rad/s)',1,4,0.1,2]],
    'Frequency (rad/s)','Loop phase (°)',([lead,wc])=>{
      const s=Math.sin(radians(lead)),alpha=(1+s)/(1-s),T=Math.sqrt(alpha)/wc,K=wc*Math.hypot(1,wc)/Math.sqrt(alpha);
      const phase=w=>-90-Math.atan(w)*180/pi;
      const corrected=w=>phase(w)+(Math.atan(w*T)-Math.atan(w*T/alpha))*180/pi;
      return {curves:[curve('With lead compensation',0.05,3*wc,corrected),curve('With scalar gain',0.05,3*wc,phase)],
        result:'At crossover, nominal margin is '+fmt(180+corrected(wc),2)+'°. The required scalar gain is '+fmt(K,3)+' and the high-frequency controller gain is '+fmt(alpha*K,3)+'.'};
    });

  register(control, 'relay-regulation',
    [['Band half-width about 21 °C (K)',0.25,2.5,0.125,1],['Heater power (W)',100,300,5,200]],
    'Time (s)','Temperature (°C)',([w,P])=>{
      const tau=200,Ta=15,eq=Ta+P/10,lo=21-w,hi=21+w;
      const on=tau*Math.log((eq-lo)/(eq-hi)),off=tau*Math.log((hi-Ta)/(lo-Ta)),period=on+off;
      const temperature=t=>{const phase=t%period;return phase<=on?eq+(lo-eq)*Math.exp(-phase/tau):Ta+(hi-Ta)*Math.exp(-(phase-on)/tau);};
      return {curves:[curve('Thermostat temperature',0,2*period,temperature,300),curve('Upper threshold',0,2*period,()=>hi),curve('Lower threshold',0,2*period,()=>lo)],
        result:'Heating takes '+fmt(on,2)+' s and cooling takes '+fmt(off,2)+' s. Duty is '+fmt(100*on/period,2)+'% and energy balance gives a mean temperature of '+fmt(Ta+P*on/(period*10),2)+' °C.'};
    });

})();
