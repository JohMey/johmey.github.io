// Chapter experiments use SI internally. The Poisson discretisation is also available as a OneSimulation input.
(() => {
  const registry = window.fundamentalStudies ||= {};
  const pi = Math.PI, mu0 = 4e-7*pi, eps0 = 8.8541878128e-12;
  const line = (name, end, f, start=0, count=160) => ({
    name, points: Array.from({length:count+1}, (_,i) => {
      const x = start+(end-start)*i/count; return [x,f(x)];
    })
  });
  function model(key, controls, xLabel, yLabel, calculate) {
    registry['em-core-'+key] = { controls,xLabel,yLabel,calculate };
  }
  model('charge',[['Charge Q (nC)',1,12,.1,4],['Cloud radius R (cm)',1,8,.1,3]],
    'Distance from centre (cm)','Electric field (kV/m)',([q,R])=>{
      const a=R/100, kq=1e-9*q/(4*pi*eps0);
      const field=x=>x===0 ? 0 : kq*Math.min(1,(x/R)**3)/(x/100)**2/1000;
      return {curves:[line('Uniform ball',16,field),line('Thin spherical shell',16,x=>x<R ? 0 : kq/(x/100)**2/1000)],
        result:'At the surface the exterior limit is '+(kq/a**2/1000).toFixed(3)+' kV/m. Halfway out, the solid ball gives '+field(R/2).toFixed(3)+' kV/m.'};
    });
  model('poisson',[['Cells per side',4,64,2,12],['Charge amplitude ρ₀ (nC/m³)',.1,10,.1,2]],
    'Position x (m), midline y = 0.5 m','Potential (V)',([n,rho])=>{
      // On a sine eigenmode conjugate gradients reaches the exact discrete answer in one step.
      const h=1/n, load=[], Au=[], index=(i,j)=>j*(n+1)+i;
      for(let j=0;j<=n;j++) for(let i=0;i<=n;i++)
        load.push(i===0||i===n||j===0||j===n ? 0 : rho*1e-9/eps0*Math.sin(pi*i*h)*Math.sin(pi*j*h));
      let norm=0,pAp=0;
      for(let j=1;j<n;j++) for(let i=1;i<n;i++) {
        const k=index(i,j);
        Au[k]=(4*load[k]-load[k-1]-load[k+1]-load[k-n-1]-load[k+n+1])/(h*h);
        norm+=load[k]**2; pAp+=load[k]*Au[k];
      }
      const alpha=norm/pAp, peak=rho*1e-9/(eps0*2*pi*pi);
      const points=Array.from({length:n+1},(_,i)=>[i*h,alpha*load[index(i,n/2)]]);
      const v=points[n/2][1];
      return {curves:[line('Continuous solution',1,x=>peak*Math.sin(pi*x)),{name:'Five-point grid solution',points}],
        result:'Centre potential '+v.toFixed(5)+' V on the grid, '+peak.toFixed(5)+' V in the continuum. Nodal error '+(v-peak).toPrecision(4)+' V.'};
    });
  registry['em-core-poisson'].localModel = {file:'grounded-square.py',args:([n,q])=>'--cells '+n+' --charge '+q};
  model('gap',[['Relative permeability',100,4000,10,1000],['Magnetomotive force NI (A turns)',10,300,1,100]],
    'Air gap (mm)','Magnetic flux density (T)',([mur,ni])=>{
      const B=g=>mu0*ni/(.20/mur+g/1000);
      return {curves:[line('Core and gap',3,B),line('Core with twice the permeability',3,g=>mu0*ni/(.20/(2*mur)+g/1000))],
        result:'At 1 mm the field is '+B(1).toFixed(4)+' T. Equal core and gap reluctances occur at '+(200/mur).toPrecision(3)+' mm.'};
    });
  model('brake',[['Magnetic field (T)',.1,1.2,.05,.5],['Circuit resistance (Ω)',.1,1,.05,.25]],
    'Time (s)','Speed (m/s)',([b,r])=>{
      const tau=.04*r/(b*.2)**2;
      return {curves:[line('Moving conductor',5,t=>2*Math.exp(-t/tau)),line('Open circuit',5,()=>2)],
        result:'Decay time '+tau.toPrecision(4)+' s. At one decay time the speed is '+(2/Math.E).toFixed(3)+' m/s and '+(100*(1-Math.exp(-2))).toFixed(1)+'% of the original kinetic energy has gone into Joule heat.'};
    });
  model('standing',[['Reflected electric amplitude / incident',-1,1,.02,.6],['Phase of incident wave (degrees)',0,360,1,30]],
    'Position x / wavelength','Energy density / reference',([r,deg])=>{
      const p=deg*pi/180, s=x=>Math.cos(2*pi*x-p), t=x=>r*Math.cos(2*pi*x+p);
      return {curves:[line('Electric part',1,x=>(s(x)+t(x))**2),line('Magnetic part',1,x=>(s(x)-t(x))**2)],
        result:'The cycle-averaged net power is '+(100*(1-r*r)).toFixed(1)+'% of the incident power. Reversing the sign of reflection swaps the pattern at the boundary.'};
    });
  model('dipole',[['Polar-cap half angle (degrees)',1,90,1,45]],'Polar angle θ (degrees)','Fraction of total power',
    ([a])=>{const fraction=d=>{const c=Math.cos(d*pi/180);return (2-3*c+c**3)/4;};
      return {curves:[line('One polar cap',180,fraction),line('Isotropic source, same total power',180,d=>(1-Math.cos(d*pi/180))/2),{name:'Selected cone',points:[[a,0],[a,fraction(a)]]}],
        result:'One polar cap collects '+(100*fraction(a)).toFixed(3)+'% of all dipole power. Both caps together collect '+(200*fraction(a)).toFixed(3)+'%.'};});
  model('multipole',[['Half separation a (cm)',.5,4,.1,2]],'Axial distance z (cm)','Potential (V)',
    ([a])=>{const p=2*8.9875517923*a*100;
      return {curves:[line('Two charges +1 and −1 nC',40,z=>p/(z*z-a*a),5),line('Dipole approximation',40,z=>p/(z*z),5)],
        result:'The dipole underestimates the exact axial potential by '+(100*(a/10)**2).toFixed(2)+'% at 10 cm. Doubling that distance divides this relative error by four.'};});
  model('sphere',[['Relative permittivity εᵣ',1,25,.1,4]],
    'Distance along positive field axis r / a','Axial field E / E₀',([er])=>{
      const k=(er-1)/(er+2), inside=3/(er+2);
      return {curves:[line('Inside',1,()=>inside),line('Outside',4,r=>1+2*k/r**3,1),line('Applied background',4,()=>1)],
        result:'Just inside, E/E₀ = '+inside.toFixed(4)+'. Just outside at the pole, E/E₀ = '+(1+2*k).toFixed(4)+'. Their ratio is '+er.toFixed(2)+'.'};});
  model('skin',[['Frequency (kHz)',1,500,1,50],['Conductivity (MS/m)',1,60,.5,58]],
    'Depth in metal (mm)','Field / field at surface',([khz,sm])=>{
      const delta=1000*Math.sqrt(1/(pi*khz*1000*mu0*sm*1e6));
      const depth=Math.max(2,3*delta);
      return {curves:[line('Magnitude',depth,x=>Math.exp(-x/delta)),line('In-phase component',depth,x=>Math.exp(-x/delta)*Math.cos(x/delta))],
        result:'Skin depth '+delta.toPrecision(4)+' mm. At this depth the magnitude has fallen to 1/e and the phase lags the surface by one radian.'};});
  model('guide',[['Guide width a (cm)',1.5,5,.1,3],['Operating frequency (GHz)',2,20,.1,8]],
    'Frequency (GHz)','Axial wave number (rad/m)',([a,f])=>{
      const c=299792458, kc=pi/(a/100), cut=c*kc/(2*pi*1e9), val=x=>(2*pi*x*1e9/c)**2-kc*kc;
      return {curves:[line('Propagation constant β',22,x=>Math.sqrt(Math.max(0,val(x))),cut),line('Decay constant α',cut,x=>Math.sqrt(Math.max(0,-val(x)))),{name:'Selected operating point',points:[[f,0],[f,Math.sqrt(Math.abs(val(f)))]]}],
        result:'Cutoff '+cut.toFixed(3)+' GHz. At '+f.toFixed(1)+' GHz the mode '+(f>=cut?'propagates with β = ':'decays with α = ')+Math.sqrt(Math.abs(val(f))).toFixed(2)+' m⁻¹.'};});
  model('line',[['Load (Ω)',0,200,1,100],['Time after launch (ns)',0,16,.1,8]],
    'Distance from source (m)','Pulse voltage (V)',([z,ns])=>{
      const g=(z-50)/(z+50), wave=t=>Math.exp(-((t-3)**2)/.65), inc=x=>wave(ns-5*x), ref=x=>g*wave(ns-5*(2-x));
      return {curves:[line('Incident',1,inc),line('Reflected',1,ref),line('Sum',1,x=>inc(x)+ref(x))],
        result:'Load reflection coefficient '+g.toFixed(3)+'. At the load the voltage is '+(inc(1)+ref(1)).toFixed(3)+' V. The matched source absorbs the return.'};});
  model('pressure',[['Reference voltage at 1 mm (V)',10,200,1,100]],
    'Plate spacing (mm)','Attractive pressure (Pa)',([v])=>{
      const P=.5*eps0*(v/.001)**2;
      return {curves:[line('Voltage maintained',3,d=>P/d**2,.4),line('Charge trapped',3,()=>P,.4)],
        result:'At 2 mm the voltage supply gives '+(P/4).toPrecision(4)+' Pa. With the original charge trapped, the pressure stays at '+P.toPrecision(4)+' Pa.'};});
})();
