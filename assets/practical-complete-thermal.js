// Quantitative companions to the twelve engineering-thermodynamics derivations.
(() => {
'use strict';
const studies=window.fundamentalStudies||={},pi=Math.PI;
const X=(a,b,n=180)=>Array.from({length:n+1},(_,i)=>a+(b-a)*i/n);
const C=(name,x,f)=>({name,points:x.map(t=>[t,f(t)])}),P=(name,x,y)=>({name,points:[[x,y]]});
const f=x=>Number(x.toPrecision(6)).toString();
const add=(k,controls,xLabel,yLabel,calculate)=>studies['complete-thermal-'+k]={controls,xLabel,yLabel,calculate};
add('spring-piston',[['Spring stiffness (N/m)',0,2000,100,1000],['Volume increase (litres)',.2,4,.1,2]],'Volume increase (litres)','Energy (kJ)',([k,d])=>{
 const volume=x=>x/1000,work=x=>100000*volume(x)+k*volume(x)**2/(2*.01**2),spring=x=>k*volume(x)**2/(2*.01**2),temp=x=>((100000+k*volume(x)/.01**2)*(.01+volume(x))-100000*.01)/(.01*287),heat=x=>.01*718*temp(x)+work(x);
 return{curves:[C('Boundary work',X(0,4),x=>work(x)/1000),C('Incremental spring energy',X(0,4),x=>spring(x)/1000),C('Required heat',X(0,4),x=>heat(x)/1000),P('Selected heat',d,heat(d)/1000)],result:'Air mass 0.010 kg, initial p=100 kPa, V=10 litres, piston area 0.010 m². Selected final pressure '+f((100000+k*volume(d)/.01**2)/1000)+' kPa; temperature rise '+f(temp(d))+' K; work '+f(work(d)/1000)+' kJ; heat '+f(heat(d)/1000)+' kJ. Quasistatic ideal gas, cv=718 J/(kg K).'};
});
add('nozzle',[['Stagnation pressure (kPa)',100,500,20,200],['Stagnation temperature (K)',200,500,10,300],['Receiver / stagnation pressure',.1,1,.05,.8]],'Receiver / stagnation pressure','Mass flux (kg/m²/s)',([p0,T0,r])=>{
 const gamma=1.4,R=287,cp=gamma*R/(gamma-1),critical=(2/(gamma+1))**(gamma/(gamma-1));
 const G=p=>{const tr=Math.max(critical,p)**((gamma-1)/gamma);return p0*1000/(R*T0)*Math.sqrt(2*cp*T0)*tr**(1/(gamma-1))*Math.sqrt(Math.max(0,1-tr));};
 return{curves:[C('Converging ideal nozzle',X(.1,1),G),P('Selected receiver',r,G(r))],result:'Critical pressure ratio '+f(critical)+'; choked mass flux '+f(G(critical))+' kg/(m² s). A 20 mm² aperture gives '+f(G(r)*.02)+' g/s at the selected receiver pressure. '+(r<=critical?'The converging exit is sonic.':'The converging exit is unchoked.')};
});
add('mixing',[['Total amount (mol)',1,10,.5,4],['First-species mole fraction',.01,.99,.01,.25],['Environment temperature (K)',250,350,5,300]],'First-species mole fraction','Generated entropy (J/K)',([N,y,T])=>{
 const s=x=>-8.314462618*N*(x*Math.log(x)+(1-x)*Math.log(1-x));
 return{curves:[C('Two distinct ideal species',X(.001,.999,300),s),P('Selected mixture',y,s(y))],result:'Entropy generation '+f(s(y))+' J/K; destroyed work potential T0 Sgen = '+f(T*s(y)/1000)+' kJ. Equal initial temperature and pressure; the same-species case has no mixing term.'};
});
add('cascade',[['Mean intermediate temperature (K)',250,290,2,270],['Interstage temperature gap (K)',0,20,1,0],['Cooling duty (kW)',1,5,.25,2]],'Mean intermediate temperature (K)','Compressor power (kW)',([Ti,g,Q])=>{
 const w1=t=>Q*((t+g/2)/240-1),w2=t=>Q*(t+g/2)/240*(300/(t-g/2)-1),total=t=>w1(t)+w2(t),sgen=Q*(Ti+g/2)/240*(1/(Ti-g/2)-1/(Ti+g/2));
 return{curves:[C('Lower-stage work',X(250,290),w1),C('Upper-stage work',X(250,290),w2),C('Total work',X(250,290),total),P('Selected total',Ti,total(Ti))],result:'Cold/hot reservoirs 240/300 K. Total work '+f(total(Ti))+' kW; COP '+f(Q/total(Ti))+'. Interstage entropy generation '+f(sgen*1000)+' W/K; 300 K × Sgen = '+f(300*sgen)+' kW equals the extra work above the reversible limit.'};
});
add('spherical-insulation',[['Insulation conductivity (W/m/K)',.02,.08,.005,.04],['Outer convection coefficient (W/m²/K)',5,50,5,10],['Selected outer radius (mm)',3,20,.5,6]],'Outer insulation radius (mm)','Heat loss for 40 K excess (W)',([k,h,r])=>{
 const resistance=x=>(1/.003-1/(x/1000))/(4*pi*k)+1/(4*pi*h*(x/1000)**2),heat=x=>40/resistance(x);
 return{curves:[C('Spherical shell + surface film',X(3,20),heat),P('Selected coating',r,heat(r))],result:'Inner radius 3 mm. Critical radius 2k/h = '+f(2000*k/h)+' mm. Bare loss '+f(heat(3))+' W; selected coated loss '+f(heat(r))+' W. The film coefficient and inner temperature are held fixed.'};
});
add('probe-ramp',[['Probe time constant (s)',1,20,1,8],['Fluid ramp rate (K/s)',-.5,1,.1,.5],['Ramp duration before hold (s)',5,40,1,16]],'Time since ramp begins (s)','Temperature (°C)',([tau,beta,hold])=>{
 const fluid=t=>20+beta*Math.min(t,hold),lagEnd=beta*tau*(-Math.expm1(-hold/tau)),probe=t=>t<=hold?20+beta*(t-tau*(-Math.expm1(-t/tau))):20+beta*hold-lagEnd*Math.exp(-(t-hold)/tau);
 return{curves:[C('Fluid ramp then hold',X(0,80,500),fluid),C('Lumped probe reading',X(0,80,500),probe)],result:'At end of ramp fluid '+f(fluid(hold))+' °C, probe '+f(probe(hold))+' °C, signed lag '+f(lagEnd)+' K. Long-ramp lag βτ='+f(beta*tau)+' K; the hold tests the same time constant independently. Internal temperature gradients are neglected.'};
});
add('turbine-quality',[['Turbine isentropic efficiency',.4,1,.02,.8],['Inlet entropy (kJ/kg/K)',6,7.4,.05,6.65]],'Turbine isentropic efficiency','Outlet enthalpy (kJ/kg)',([eta,s])=>{
 const x=(s-.649)/7.5,hs=192+x*2393,h=e=>3200-e*(3200-hs),quality=(h(eta)-192)/2393;
 return{curves:[C('Actual outlet enthalpy',X(.4,1),h),C('Saturated-vapour bound',X(.4,1),()=>2585),C('Saturated-liquid bound',X(.4,1),()=>192),P('Selected efficiency',eta,h(eta))],result:'Isentropic outlet quality '+f(x)+'; selected work '+f(3200-h(eta))+' kJ/kg. '+(quality>1?'Actual outlet exceeds saturated-vapour enthalpy; it is superheated and no mixture quality is assigned.':'Actual mixture quality '+f(quality)+'.')+' Rounded outlet saturation properties are fixed as in the derivation.'};
});
add('coil-bypass',[['Bypass fraction inferred from humidity',0,1,.05,.25],['Measured outlet enthalpy (kJ/kg dry air)',30,65,.025,39.725]],'Bypass fraction','Moist-air enthalpy (kJ/kg dry air)',([b,measured])=>{
 const h=x=>29.3+x*(71-29.3),w=.0076+b*(.016-.0076),mb=1.2*(.016-w),Q=1.2*(71-h(b))-mb*42,fromH=(measured-29.3)/(71-29.3);
 return{curves:[C('Single surface-state model',X(0,1),h),P('Measured humidity and enthalpy',b,measured),P('Consistent enthalpy at humidity bypass',b,h(b))],result:'Humidity predicts ωout='+f(w)+' kg/kg and hout='+f(h(b))+' kJ/kg; enthalpy measurement implies b='+f(fromH)+'. Discrepancy '+f(measured-h(b))+' kJ/kg. At the model state and 1.2 kg/s dry air: condensate '+f(mb)+' kg/s; heat removal '+f(Q)+' kW including liquid enthalpy 42 kJ/kg.'};
});
add('flow-exergy',[['Inlet temperature (K)',250,500,10,360],['Selected pressure / ambient pressure',1,6,.1,3]],'Inlet pressure / ambient pressure','Specific physical exergy (kJ/kg)',([T,p])=>{
 const thermal=1.005*(T-300-300*Math.log(T/300)),pressure=r=>.287*300*Math.log(r);
 return{curves:[C('Pressure contribution',X(1,6),pressure),C('Thermal contribution',X(1,6),()=>thermal),C('Total physical flow exergy',X(1,6),r=>thermal+pressure(r)),P('Selected inlet',p,thermal+pressure(p))],result:'Environment 300 K, 100 kPa. Thermal '+f(thermal)+', pressure '+f(pressure(p))+', total '+f(thermal+pressure(p))+' kJ/kg. At ambient temperature an ideal-gas throttle from this pressure to ambient destroys '+f(.1*pressure(p))+' kW at 0.1 kg/s without changing enthalpy.'};
});
add('condenser-calorimetry',[['Cooling-water mass rate (kg/s)',300,700,25,500],['Measured water temperature rise (K)',5,14,.25,10]],'Cooling-water temperature rise (K)','Inferred exhaust quality (where physically admissible)',([mw,dT])=>{
 const q=t=>mw*4.18*t,x=t=>q(t)/(10*2393),limit=10*2393/(mw*4.18),value=x(dT);
 return{curves:[C('Latent-heat interpolation',X(5,14),x),C('Dry-saturated upper bound',X(5,14),()=>1),P('Selected reading',dT,value)],result:'10 kg/s steam; latent heat 2393 kJ/kg. Condenser duty '+f(q(dT)/1000)+' MW. '+(value>1?'The inferred ratio exceeds one, so a saturated-mixture interpretation is inconsistent; check measurements or superheat.':'Quality '+f(value)+'; ordinary-cycle net power '+f(30-q(dT)/1000)+' MW for 30 MW boiler heat.')+' Greatest rise compatible with saturated exhaust '+f(limit)+' K.'};
});
add('lmtd-area',[['Selected first terminal difference (K)',1,50,1,30],['Second terminal difference (K)',5,50,1,35],['Overall coefficient (W/m²/K)',200,1000,50,500]],'First terminal temperature difference (K)','Area for 70 kW duty (m²)',([a,b,U])=>{
 const mean=x=>Math.abs(x-b)<1e-8?b:(x-b)/Math.log(x/b),area=x=>70000/(U*mean(x));
 return{curves:[C('Logarithmic-mean sizing',X(1,50,300),area),C('Arithmetic-mean estimate',X(1,50),x=>70000/(U*(x+b)/2)),P('Selected design',a,area(a))],result:'LMTD '+f(mean(a))+' K; required area '+f(area(a))+' m². Equal terminal differences use the continuous limit. Positive paired approaches at each physical end are required.'};
});
add('radiation-heater',[['Surface emissivity',.2,1,.05,.8],['Convection coefficient (W/m²/K)',2,20,1,8],['Selected surface temperature (K)',310,450,5,350]],'Surface temperature (K)','Required heater power over 0.010 m² (W)',([eps,h,T])=>{
 const sigma=5.670374419e-8,rad=t=>eps*sigma*(t**4-300**4),conv=t=>h*(t-310),power=t=>.01*(rad(t)+conv(t)),slope=.01*(h+4*eps*sigma*T**3),increment=.1/slope;
 return{curves:[C('Exact radiation + convection balance',X(300,470),power),C('Tangent at selected operating point',X(300,470),t=>power(T)+slope*(t-T)),P('Selected operating point',T,power(T))],result:'Enclosure 300 K, gas 310 K. Heater '+f(power(T))+' W; secant radiation coefficient '+f(eps*sigma*(T+300)*(T*T+300**2))+' W/(m² K); tangent '+f(4*eps*sigma*T**3)+' W/(m² K). Additional 0.100 W gives linearised ΔT='+f(increment)+' K.'};
});
})();
