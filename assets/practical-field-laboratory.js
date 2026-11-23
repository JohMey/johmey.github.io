// Field experiments, with a reference independent of the numerical path in each exercise.
(() => {
  const studies = window.fundamentalStudies ||= {};
  const pi = Math.PI;
  const grid = (a,b,n) => Array.from({length:n+1},(_,i)=>a+(b-a)*i/n);
  const curve = (name,x,y) => ({name,points:x.map((value,i)=>[value,y(value,i)])});
  const show = x => String(Number(x.toPrecision(5)));
  const local = name => ({file:'field-laboratory.py',args:([n,g])=>name+' --cells '+n+' --contrast '+g});
  function add(key,controls,xLabel,yLabel,calculate,localModel) {
    studies['field-lab-'+key]={controls,xLabel,yLabel,calculate,localModel};
  }
  // Store just the interior unknowns; missing neighbours have fixed zero Dirichlet values.
  function squarePoisson(n,g) {
    const side=n-1, u=new Float64Array(side*side), r=new Float64Array(u.length);
    for(let j=1;j<n;j++) for(let i=1;i<n;i++) {
      r[(j-1)*side+i-1]=pi*pi*(2*Math.sin(pi*i/n)+5*g*Math.sin(2*pi*i/n))*Math.sin(pi*j/n);
    }
    const dot=(a,b)=>a.reduce((sum,v,i)=>sum+v*b[i],0);
    const p=r.slice(), Ap=new Float64Array(u.length);let rr=dot(r,r),initial=rr,iterations=0;
    while(rr>initial*1e-22 && iterations<30) {
      for(let j=0;j<side;j++)for(let i=0;i<side;i++){
        const k=j*side+i;
        Ap[k]=n*n*(4*p[k]-(i?p[k-1]:0)-(i+1<side?p[k+1]:0)-(j?p[k-side]:0)-(j+1<side?p[k+side]:0));
      }
      const alpha=rr/dot(p,Ap);
      for(let k=0;k<p.length;k++){u[k]+=alpha*p[k];r[k]-=alpha*Ap[k];}
      const next=dot(r,r);
      for(let k=0;k<p.length;k++)p[k]=r[k]+(next/rr)*p[k];
      rr=next;iterations++;
    }
    return {mid:grid(0,1,n).map((x,i)=>i===0||i===n?0:u[(n/2-1)*side+i-1]),rel:Math.sqrt(rr/initial),iterations};
  }
  add('two-mode-verification',[['Cells per side',8,48,4,16],['Signed second-mode amplitude',-2,2,.05,.5]],
    'Midline x (m)','Potential (V)',([n,g])=>{
      const x=grid(0,1,n),z=squarePoisson(n,g),exact=t=>Math.sin(pi*t)+g*Math.sin(2*pi*t);
      const ec=Math.max(...x.map((t,i)=>Math.abs(exact(t)-z.mid[i])));
      return {curves:[curve('Continuous field',grid(0,1,200),exact),curve('Solved grid field',x,(_,i)=>z.mid[i])],
        result:'Midline maximum error '+show(ec)+' V. Quarter-point difference '+show(z.mid[n/4]-z.mid[3*n/4])+' V. '+z.iterations+' CG step(s), relative residual '+z.rel.toExponential(2)+'.'};
    },local('modes'));
  add('graded-capacitor',[['Cells across gap',4,64,2,16],['Permittivity increase g',0,12,.1,3]],
    'Position x / gap','Potential (V)',([n,g])=>{
      const x=grid(0,1,n),drops=x.slice(1).map((_,i)=>1/(n*(1+g*(i+.5)/n)));
      const R=drops.reduce((a,b)=>a+b,0),v=[0];for(const d of drops)v.push(v.at(-1)+d/R);
      const exact=t=>g===0?t:Math.log1p(g*t)/Math.log1p(g);
      return {curves:[curve('Analytical voltage',grid(0,1,160),exact),curve('Flux balance',x,(_,i)=>v[i])],
        result:'Effective relative permittivity '+show(1/R)+' on the grid and '+show(g===0?1:g/Math.log1p(g))+' analytically. Midpoint voltage '+show(v[n/2])+' V.'};
    },local('capacitor'));
  add('screening-boundary-layer',[['Cells across slab',4,64,2,16],['Gap / screening length',0,20,.25,4]],
    'Position x / slab thickness','Potential / boundary voltage',([n,g])=>{
      const x=grid(0,1,n),diag=new Float64Array(n-1).fill(2+(g/n)**2),rhs=new Float64Array(n-1);rhs[0]=1;
      for(let i=1;i<n-1;i++){diag[i]-=1/diag[i-1];rhs[i]+=rhs[i-1]/diag[i-1];}
      const u=new Float64Array(n+1);u[0]=1;
      for(let j=n-1;j>=1;j--)u[j]=(rhs[j-1]+u[j+1])/diag[j-1];
      const exact=t=>g===0?1-t:Math.sinh(g*(1-t))/Math.sinh(g);
      return {curves:[curve('Continuum slab',grid(0,1,160),exact),curve('Solved grid values',x,(_,i)=>u[i])],
        result:'Midpoint '+show(u[n/2])+', exact '+show(exact(.5))+'. Discrete decay constant × thickness = '+show(2*n*Math.asinh(g/(2*n)))+'.'};
    },local('screen'));
  add('regularised-charge',[['Charge-cloud radius a (mm)',.5,8,.1,3],['Probe radius r (mm)',.2,24,.2,6]],
    'Distance from centre (mm)','Flux charge / total charge',([a,r])=>{
      const x=grid(0,24,240),fraction=t=>t**3/(t*t+a*a)**1.5;
      return {curves:[curve('Softened charge distribution',x,fraction),curve('Point charge (outside centre)',x.slice(1),()=>1),{name:'Probe',points:[[r,0],[r,fraction(r)]]}],
        result:'The probe encloses '+show(100*fraction(r))+'% of the charge. Reducing a by half gives '+show(100*r**3/(r*r+a*a/4)**1.5)+'% at the same physical probe.'};
    });
  add('discrete-wave',[['Points per wavelength',5,50,1,12],['Courant number',.1,1,.05,.8]],
    'Position along ring (wavelengths)','Electric field / initial amplitude',([n,c])=>{
      const theta=2*pi/n, wt=2*Math.asin(c*Math.sin(theta/2)),steps=Math.round(5*n/c),time=steps*c/n,
        x=grid(0,1,n).slice(0,-1);let old=x.map(t=>Math.sin(2*pi*t+wt)),u=x.map(t=>Math.sin(2*pi*t));
      for(let step=0;step<steps;step++){const next=u.map((v,i)=>2*v-old[i]+c*c*(u[(i+1)%n]-2*v+u[(i+n-1)%n]));old=u;u=next;}
      return {curves:[curve('Continuous wave',grid(0,1,160),t=>Math.sin(2*pi*(t-time))),curve('Time-stepped nodes',grid(0,1,n),(_,j)=>u[j%n])],
        result:steps+' steps, '+show(time)+' physical periods. Phase speed ratio '+show(wt/(c*theta))+'. Accumulated phase lag '+show((c*theta-wt)*steps*180/pi)+'°.'};
    });
  add('deembedding-a-line',[['Load resistance (Ω)',10,180,1,75],['Probe plane from load (wavelengths)',0,1,.01,.13]],
    'Distance from load (wavelengths)','Voltage magnitude / incident amplitude',([load,d])=>{
      const z0=50,r=(load-z0)/(load+z0), mag=s=>Math.sqrt(1+r*r+2*r*Math.cos(4*pi*s)), xs=grid(0,1,180);
      const phase=(r>=0?0:180)-720*d, principal=(phase+900)%360-180;
      return {curves:[curve('Standing-wave envelope',xs,mag),{name:'Selected reference plane',points:[[d,0],[d,mag(d)]]}],
        result:'Load Γ = '+show(r)+'. Probe-plane phase '+(Math.abs(r)<1e-14?'undefined (matched load)':show(principal)+'°')+
          '. Voltage standing-wave ratio '+show((1+Math.abs(r))/(1-Math.abs(r)))+'.'};
    });
  add('conducting-film',[['Thickness / skin depth',.12,6,.02,1],['Surface current (A/m, peak)',1,30,1,10]],
    'Depth / skin depth','Magnetic field / front field',([d,h])=>{
      // |sinh((1+i)t)|² = sinh²(t)+sin²(t); the back face H=0.
      const A=t=>Math.hypot(Math.sinh(t),Math.sin(t)),x=grid(0,d,160);
      const den=Math.cosh(2*d)-Math.cos(2*d),R=(Math.sinh(2*d)+Math.sin(2*d))/den,
        X=(Math.sinh(2*d)-Math.sin(2*d))/den;
      return {curves:[curve('Finite film |H|',x,t=>A(d-t)/A(d)),curve('Thick conductor approximation',x,t=>Math.exp(-t))],
        result:'Normalised surface impedance '+show(R)+' + i '+show(X)+'. Mean power per area divided by the thick-conductor sheet resistance is '+show(.5*h*h*R)+' A²/m².'};
    });
  add('aperture-sampling',[['Gaussian aperture radius w (mm)',.4,2,.05,1],['Window half-width / w',.75,4,.05,2],['Samples across window',8,80,2,30]],
    'Transverse spatial frequency (cycles/mm)','Normalised field amplitude',([w,a,n])=>{
      const dx=2*a*w/n,j=grid(0,n-1,n-1).map(i=>-a*w+(i+.5)*dx),weights=j.map(t=>Math.exp(-((t/w)**2))),norm=weights.reduce((s,v)=>s+v,0);
      const measured=f=>weights.reduce((s,v,i)=>s+v*Math.cos(2*pi*f*j[i]),0)/norm,x=grid(0,2,220);
      return {curves:[curve('Infinite Gaussian',x,f=>Math.exp(-((pi*w*f)**2))),curve('Sampled finite window',x,measured)],
        result:'Sample spacing '+show(dx)+' mm, Nyquist frequency '+show(.5/dx)+' cycles/mm. Measured amplitude at 1 cycle/mm: '+show(measured(1))+'.'};
    });
  add('ringdown-estimation',[['Additive detector offset / initial energy',0,.2,.005,.04],['Start of fitting window (lifetimes)',0,3,.05,1]],
    'Time / true energy lifetime','Natural log of normalised signal',([b,t0])=>{
      const y=t=>Math.log(Math.exp(-t)+b),t1=t0+1,rate=y(t0)-y(t1),x=grid(0,5,180);
      return {curves:[curve('Ideal log energy',x,t=>-t),curve('Detector with offset',x,y),{name:'Fit over one lifetime',points:[[t0,y(t0)],[t1,y(t1)]]}],
        result:'Two-point inferred lifetime '+show(1/rate)+' times the true lifetime. The same fractional bias transfers to Q for a known mode frequency.'};
    });
  add('cyclotron-integration',[['Steps per exact orbit',8,80,1,24],['Number of orbits',1,12,1,4]],
    'Time (exact periods)','Speed / initial speed',([n,orbits])=>{
      const d=2*pi/n,t=d/2,c=(1-t*t)/(1+t*t),s=2*t/(1+t*t);let ex=1,ey=0,bx=1,by=0;
      const eu=[[0,1]],bo=[[0,1]];
      for(let j=1;j<=n*orbits;j++) {const x=ex;ex=x+d*ey;ey-=d*x;
        const a=bx;bx=c*a+s*by;by=-s*a+c*by;
        eu.push([j/n,Math.hypot(ex,ey)]);bo.push([j/n,Math.hypot(bx,by)]);}
      return {curves:[{name:'Forward Euler',points:eu},{name:'Cayley / magnetic Boris rotation',points:bo}],
        result:'After '+orbits+' orbit(s), Euler speed '+show(Math.hypot(ex,ey))+', rotation speed '+show(Math.hypot(bx,by))+
          '. Rotation still lags the exact phase by '+show((d-2*Math.atan(t))*n*orbits*180/pi)+'°.'};
    });
  add('differentiating-radiation',[['Samples per physical period',12,96,2,32],['Position disturbance / amplitude',0,.01,.00025,.002]],
    'Time (periods)','Acceleration / reference peak',([n,eps])=>{
      const dt=2*pi/n,position=j=>Math.cos(j*dt)+eps*(j%2===0?1:-1),xs=grid(0,1,n);
      const acc=xs.map((_,j)=>(position(j-1)-2*position(j)+position(j+1))/(dt*dt)),ratio=2*acc.slice(0,n).reduce((sum,v)=>sum+v*v,0)/n;
      return {curves:[curve('Reference acceleration',xs,t=>-Math.cos(2*pi*t)),curve('Second difference',xs,(_,j)=>acc[j])],
        result:'Mean squared acceleration / exact value '+show(ratio)+'. The alternating error contributes peak acceleration '+show(4*eps/(dt*dt))+' in the same units.'};
    });
  add('identify-a-coil',[['Probe frequency (Hz)',5,800,5,100],['Coil inductance (mH)',5,160,1,40]],
    'Frequency (Hz)','Impedance (Ω)',([f,mh])=>{
      const R=20,L=mh*.001,x=grid(5,800,159),mag=t=>Math.hypot(R,2*pi*t*L),phase=Math.atan2(2*pi*f*L,R)*180/pi;
      return {curves:[curve('Impedance magnitude',x,mag),curve('Resistive part',x,()=>R),curve('Inductive reactance',x,t=>2*pi*t*L),{name:'Probe',points:[[f,0],[f,mag(f)]]}],
        result:'At the probe: |Z| = '+show(mag(f))+' Ω, phase '+show(phase)+'°. Equal resistive and reactive contributions at '+show(R/(2*pi*L))+' Hz.'};
    });
  add('relaxation-spectroscopy',[['Slow fraction of total dielectric strength',0,1,.02,.5],['Slow / fast relaxation time',1,100,1,12]],
    'log₁₀ (angular frequency × fast time)','Relative permittivity component',([fraction,ratio])=>{
      const xs=grid(-3,2,200);
      const evalAt=(t,loss)=>{
        const x=10**t,y=x*ratio;
        return (loss?0:2)+6*((1-fraction)*(loss?x:1)/(1+x*x)+fraction*(loss?y:1)/(1+y*y));
      };
      const peak=xs.reduce((a,b)=>evalAt(a,true)>evalAt(b,true)?a:b);
      return {curves:[curve('Real part ε′',xs,x=>evalAt(x,false)),curve('Loss ε″',xs,x=>evalAt(x,true))],
        result:'Largest sampled loss '+show(evalAt(peak,true))+' near ωτ_fast = '+show(10**peak)+'. Relative permittivity tends to 8 at low frequency and 2 at high frequency.'};
    });
  add('hall-measurement',[['Electron density (10²² m⁻³)',.2,5,.1,1],['Current (mA)',1,12,.25,5]],
    'Applied field B (T)','Transverse voltage (mV)',([n,i])=>{
      const x=grid(-.5,.5,180),hall=b=>-i*1e-3*b/(n*1e22*1.602176634e-19*.0005)*1e3;
      return {curves:[curve('Hall component',x,hall),curve('Reading with 4.3 mV pickup',x,b=>4.3+hall(b))],
        result:'At +0.2 T the instrument reads '+show(4.3+hall(.2))+' mV; at −0.2 T it reads '+show(4.3+hall(-.2))+
          ' mV. Half-difference '+show(hall(.2))+' mV.'};
    });
  add('thermal-noise-and-bandwidth',[['Resistance (kΩ)',1,50,1,10],['Temperature (K)',100,400,5,300],['Detector bandwidth (kHz)',5,500,5,100]],
    'Detector upper cutoff (kHz)','Collected voltage noise (μV RMS)',([k,T,b])=>{
      const R=1000*k,C=100e-12,kb=1.380649e-23,corner=1/(2*pi*R*C)/1000,total=1e6*Math.sqrt(kb*T/C);
      const rms=f=>total*Math.sqrt(2/pi*Math.atan(f/corner)), x=grid(0,700,210);
      return {curves:[curve('Finite bandwidth',x,rms),curve('Equilibrium total',x,()=>total),{name:'Chosen bandwidth',points:[[b,0],[b,rms(b)]]}],
        result:'RC corner '+show(corner)+' kHz. Detector reads '+show(rms(b))+' μV RMS, collecting '+show(100*(rms(b)/total)**2)+'% of equilibrium variance.'};
    });
  add('polarisation-and-stability',[['Temperature coefficient s',-2,2,.05,-1],['Applied field e',-.8,.8,.02,.18]],
    'Polarisation p','Field-controlled potential g(p)',([s,e])=>{
      const value=p=>p**4/4+s*p*p/2-e*p, xs=grid(-2.1,2.1,210),disc=e*e/4+(s/3)**3;
      let roots;
      if(disc>1e-13)roots=[Math.cbrt(e/2+Math.sqrt(disc))+Math.cbrt(e/2-Math.sqrt(disc))];
      else if(disc>=-1e-13){const t=Math.cbrt(e/2);roots=Math.abs(t)<1e-10?[0]:[2*t,-t];}
      else {const phi=Math.acos(Math.max(-1,Math.min(1,e/(2*Math.sqrt(-((s/3)**3))))));roots=[0,1,2].map(j=>2*Math.sqrt(-s/3)*Math.cos((phi+2*pi*j)/3));}
      roots.sort((a,b)=>a-b);
      return {curves:[curve('Potential',xs,value),...roots.map(p=>({name:'Stationary p = '+show(p),points:[[p,value(p)]]}))],
        result:'Stationary polarisations: '+roots.map(p=>show(p)+' ('+(3*p*p+s>1e-7?'minimum':3*p*p+s< -1e-7?'maximum':s===0&&e===0?'flat minimum':'zero curvature')+')').join(', ')+'.'};
    });
  add('london-penetration',[['Slab thickness d (nm)',20,400,5,100],['London length λ (nm)',20,160,2,50]],
    'Fractional position x / thickness','Magnetic field / applied field',([d,lambda])=>{
      const t=d/(2*lambda),field=x=>Math.cosh(2*t*x)/Math.cosh(t),avg=Math.tanh(t)/t,x=grid(-.5,.5,150);
      return {curves:[curve('Slab field',x,field),curve('Mean field',x,()=>avg),curve('Unscreened field',x,()=>1)],
        result:'At d/λ = '+show(d/lambda)+' the centre retains '+show(100*field(0))+'% of the applied field and the thickness average '+show(100*avg)+'%.'};
    });
  add('bloch-optics',[['High index n₂ (n₁ = 1.5)',1.5,3,.05,2],['Probe frequency / design frequency',.5,1.5,.01,1]],
    'Angular frequency / design frequency','Half trace of transfer matrix',([n,f])=>{
      const r=n/1.5,trace=x=>Math.cos(pi*x/2)**2-.5*(r+1/r)*Math.sin(pi*x/2)**2,x=grid(0,2,240),t=trace(f),alpha=Math.acosh(Math.max(1,Math.abs(t)));
      return {curves:[curve('Half trace',x,trace),curve('Upper propagation limit',x,()=>1),curve('Lower propagation limit',x,()=>-1),{name:'Chosen probe',points:[[f,t]]}],
        result:'First gap width / design frequency '+show(4/pi*Math.asin((n-1.5)/(n+1.5)))+'. Probe half trace '+show(t)+'. Attenuation per cell '+show(alpha)+'.'};
    });
  add('effective-tensors',[['Fraction of low-permittivity phase',0,1,.02,.5],['Measurement angle (degrees)',0,90,1,45]],
    'Angle to layer normal (degrees)','Apparent relative permittivity',([f,deg])=>{
      const t=2*f+10*(1-f),n=1/(f/2+(1-f)/10),val=(a,fixedE)=>{const u=Math.cos(a*pi/180)**2,v=1-u;return fixedE?n*u+t*v:1/(u/n+v/t);},x=grid(0,90,180);
      return {curves:[curve('Fixed field direction',x,a=>val(a,true)),curve('Fixed displacement direction',x,a=>val(a,false)),{name:'Selected angle',points:[[deg,val(deg,false)],[deg,val(deg,true)]]}],
        result:'Principal relative permittivities: normal '+show(n)+', tangential '+show(t)+'. At '+deg+'° the two responses are '+show(val(deg,true))+' and '+show(val(deg,false))+'.'};
    });
  add('faraday-rotation',[['Index difference (10⁻⁴)',0,12,.1,5],['Sample length (mm)',.02,1,.02,.2]],
    'Analyser angle in laboratory axes (degrees)','Transmitted fraction',([dn,mm])=>{
      const theta=pi*mm*1e-3*(dn*1e-4)/(600e-9),x=grid(0,180,180),intensity=(d,t)=>Math.cos(d*pi/180-t)**2;
      return {curves:[curve('Positive splitting',x,d=>intensity(d,theta)),curve('Reversed splitting',x,d=>intensity(d,-theta))],
        result:'Unwrapped single-pass rotation '+show(theta*180/pi)+'°. At a 45° analyser the two transmitted fractions are '+show(intensity(45,theta))+' and '+show(intensity(45,-theta))+'.'};
    });
  add('surface-polaritons',[['Dielectric relative permittivity',1,5,.05,1],['Selected frequency / surface resonance',.15,.95,.01,.6]],
    'Frequency / bulk plasma frequency','Parallel wavenumber βc/ωp',([ed,ratio])=>{
      const end=1/Math.sqrt(1+ed), xx=grid(.03*end,.96*end,150);
      const beta=w=>{const em=1-1/(w*w);return w*Math.sqrt(ed*em/(ed+em));};
      const w=ratio*end;
      return {curves:[curve('Bound TM surface mode',xx,beta),curve('Dielectric light line',xx,v=>v*Math.sqrt(ed)),
        {name:'Selected frequency',points:[[w,beta(w)]]}],
        result:'Surface resonance ωsp/ωp = '+show(end)+'. At the selected frequency β/k₀ = '+show(beta(w)/w)+'.'};
    });
  add('charge-counting-noise',[['Fano factor',.2,1.8,.02,1],['Feedback resistance (MΩ)',.1,4,.05,1]],
    'Mean photocurrent (µA)','Current noise density (pA²/Hz)',([f,r])=>{
      const xx=grid(0,1.2,80), floor=4*1.380649e-23*300/(r*1e6)*1e24;
      const shot=i=>2*1.602176634e-19*i*1e-6*f*1e24;
      return {curves:[curve('Combined variance density',xx,i=>floor+shot(i)),curve('Counting contribution',xx,shot),curve('Resistor floor',xx,()=>floor)],
        result:'The two noise contributions are equal at '+show(1000*floor/shot(1))+' nA. The zero-current density is '+show(floor)+' pA²/Hz.'};
    });
  add('gated-sheet-waves',[['Gate separation (nm)',10,250,2,50],['Selected wavelength (µm)',1,12,.1,5]],
    'Angular wavenumber (µm⁻¹)','Collective frequency (GHz)',([dn,lambda])=>{
      const d=dn*1e-9, a=2e15*(1.602176634e-19)**2/(2*12.9*8.8541878128e-12*.067*9.1093837e-31);
      const xx=grid(0,6.5,150), absolute=q=>Math.sqrt(a*q*1e6)/(2*pi*1e9);
      const gated=q=>absolute(q)*Math.sqrt(-Math.expm1(-2*q*1e6*d)), w=2*pi/lambda;
      return {curves:[curve('Free sheet',xx,absolute),curve('Grounded gate',xx,gated),
        curve('Long-wave tangent',xx,q=>q*1e6*Math.sqrt(2*a*d)/(2*pi*1e9)),{name:'Selected wavelength',points:[[w,gated(w)]]}],
        result:'At the chosen wavelength qd = '+show(w*1e6*d)+'. Gated mode '+show(gated(w))+' GHz. Free sheet '+show(absolute(w))+' GHz.'};
    });
  add('dielectric-mixtures',[['Particle relative permittivity',1,14,.1,8],['Selected volume fraction',0,.35,.01,.15]],
    'Inclusion volume fraction','Effective relative permittivity',([s,f])=>{
      const h=2, b=(s-h)/(s+2*h), xx=grid(0,.35,100), mg=v=>h*(1+2*v*b)/(1-v*b);
      return {curves:[curve('Spherical local-field approximation',xx,mg),curve('Layers, tangential field',xx,v=>(1-v)*h+v*s),
        curve('Layers, normal field',xx,v=>1/((1-v)/h+v/s)),{name:'Selected fraction',points:[[f,mg(f)]]}],
        result:'At the selected fraction εeff = '+show(mg(f))+'. The first-order dilute estimate gives '+show(h*(1+3*f*b))+'.'};
    });
})();
