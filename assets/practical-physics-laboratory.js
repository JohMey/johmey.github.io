(() => {
  "use strict";
  const studies=window.fundamentalStudies ||= {};
  const pi=Math.PI;
  const display=x=>Number(x.toPrecision(5)).toString();
  const samples=(from,to,n=150)=>Array.from({length:n+1},(_,i)=>from+i*(to-from)/n);
  const line=(name,x,fn)=>({name,points:x.map(v=>[v,fn(v)])});
  const point=(name,x,y)=>({name,points:[[x,y]]});
  function model(name,controls,xLabel,yLabel,calculate,localModel) {
    studies["physics-lab-"+name]={controls,xLabel,yLabel,calculate,localModel};
  }
  const local=args=>({file:"spectra-and-orbits.py",args});
  model("counting-limit",[["Independent sites N",4,180,1,20],["Excitation probability p",.01,.99,.01,.2]],
    "Number of recorded excitations","Probability in an integer bin",([n,p])=>{
      const peak=Math.floor((n+1)*p), weights=Array(n+1).fill(1);
      for(let k=peak;k<n;k++) weights[k+1]=weights[k]*(n-k)*p/((k+1)*(1-p));
      for(let k=peak;k>0;k--) weights[k-1]=weights[k]*k*(1-p)/((n-k+1)*p);
      const total=weights.reduce((a,b)=>a+b,0), mu=n*p, sigma=Math.sqrt(n*p*(1-p));
      const gaussian=x=>Math.exp(-.5*((x-mu)/sigma)**2)/(sigma*Math.sqrt(2*pi));
      // Simpson integration over each detector bin preserves the distinction between density and mass.
      const xs=Array.from({length:n+1},(_,k)=>k);
      // Narrow count distributions need smaller steps even though every bin is one count wide.
      const slices=2*Math.max(12,Math.ceil(16/sigma));
      const bins=xs.map(k=>{
        const left=k-.5;
        let sum=gaussian(left)+gaussian(left+1);
        for(let i=1;i<slices;i++) sum+=(i%2?4:2)*gaussian(left+i/slices);
        return sum/(3*slices);
      });
      const errors=bins.map((mass,k)=>Math.abs(mass-weights[k]/total)), worst=errors.indexOf(Math.max(...errors));
      return {curves:[line("Exact binomial count",xs,k=>weights[k]/total),line("Normal probability in the bin",xs,k=>bins[k])],
        result:"Mean "+display(mu)+", standard deviation "+display(sigma)+". Largest absolute bin error "+display(errors[worst])+" at count "+worst+". Skewness "+display((1-2*p)/sigma)+"."};
    });
  function finitePeriod(theta,length) {
    let a=1,b=Math.cos(theta*pi/360);
    while(a-b>1e-14) [a,b]=[(a+b)/2,Math.sqrt(a*b)];
    return 2*pi*Math.sqrt(length/9.81)/a;
  }
  model("pendulum-gravimeter",[["Release angle (degrees)",2,125,1,30],["Pendulum length (m)",.2,1.8,.02,1]],"Release angle (degrees)","Period (s)",([angle,l])=>{
    const xx=samples(0,125), small=2*pi*Math.sqrt(l/9.81), exact=a=>finitePeriod(a,l), t=exact(angle);
    return {curves:[line("Exact energy integral",xx,exact),line("First amplitude correction",xx,a=>small*(1+(a*pi/180)**2/16)),line("Small-angle period",xx,()=>small),point("Selected release",angle,t)],
      result:"Period "+display(t)+" s. Small-angle analysis would infer g = "+display(4*pi*pi*l/(t*t))+" m/s²."};
  });
  model("reversible-pendulum",[["Radius of gyration (m)",.12,.40,.005,.30],["Second pivot distance (m)",.06,.85,.005,.45]],"Second distance b (m)","Small-angle period (s)",([rho,b])=>{
    const a=.20, xx=samples(.06,.85), period=x=>2*pi*Math.sqrt((x+rho*rho/x)/9.81);
    return {curves:[line("Second pivot",xx,period),line("First pivot",xx,()=>period(a)),point("Selected second pivot",b,period(b))],
      result:"First period "+display(period(a))+" s; second "+display(period(b))+" s. The other matching distance is "+display(rho*rho/a)+" m."};
  });
  model("orbit-precession",[["Relative force correction δ",-.15,.15,.002,.04],["Angular grid step (degrees)",2,40,1,24]],"Nominal revolutions of azimuth","Accumulated radial phase excess (degrees)",([delta,step])=>{
    const h=step*pi/180, nu=Math.sqrt(1+delta), discrete=2*Math.asin(h*nu/2)/h, xx=samples(0,15);
    return {curves:[line("Physical radial phase",xx,r=>360*r*(nu-1)),line("Discrete radial phase",xx,r=>360*r*(discrete-1))],
      result:"Apsidal advance per radial period: exact "+display(360*(1/nu-1))+"°, numerical "+display(360*(1/discrete-1))+"°."};
  });
  function lowWell(d,t) { return -2*t*t/(d+Math.hypot(d,2*t)); }
  model("double-well-splitting",[["Middle site energy V (meV)",2,12,.1,4],["Hopping t (meV)",.5,1.5,.05,1]],"Barrier height V/t","Low-doublet splitting ΔE / t",([v,t])=>{
    const xx=samples(1.25,24), split=x=>lowWell(x+1,1)-lowWell(x-1,1), delta=split(v/t)*t;
    return {curves:[line("Exact parity sectors",xx,split),line("Three-hop approximation",xx,x=>2/(x*x)),point("Selected barrier",v/t,delta/t)],
      result:"Splitting "+display(delta)+" meV. First transfer between dressed end states at "+display(pi*.6582119569/delta)+" ps."};
  },local(([v,t])=>"wells --barrier "+v+" --hopping "+t));
  model("bell-visibility",[["Preparation visibility",0,1,.01,.80],["Symmetric analyser angle (degrees)",0,90,1,45]],"Analyser angle (degrees)","Absolute CHSH statistic",([visibility,theta])=>{
    const xx=samples(0,90), value=a=>2*visibility*(Math.cos(a*pi/180)+Math.sin(a*pi/180));
    return {curves:[line("Singlet with unpolarised admixture",xx,value),line("Local binary assignment bound",xx,()=>2),point("Selected angles",theta,value(theta))],
      result:"Predicted |S| = "+display(value(theta))+". With these angles the visibility required to reach two is "+display(1/(Math.cos(theta*pi/180)+Math.sin(theta*pi/180)))+"."};
  });
  model("ramsey-contrast",[["Mean cyclic detuning (Hz)",-1100,1100,10,500],["Shot-to-shot rms spread (Hz)",0,500,5,80]],"Free evolution (ms)","Detection probability",([frequency,spread])=>{
    const xx=samples(0,4,320), contrast=t=>Math.exp(-t/2-.5*(2*pi*spread*t/1000)**2);
    const p=(t,phase)=>(1+contrast(t)*Math.cos(2*pi*frequency*t/1000-phase))/2;
    return {curves:[line("In phase",xx,t=>p(t,0)),line("Quadrature",xx,t=>p(t,pi/2)),line("Upper envelope",xx,t=>(1+contrast(t))/2)],
      result:"Contrast after 1 ms = "+display(contrast(1))+". The two probabilities at 0.50 ms are "+display(p(.5,0))+" and "+display(p(.5,pi/2))+"."};
  });
  function chainModes(temp,bias) {
    const a=Math.exp((1+bias)/temp), d=Math.exp(1/temp)+1, b=Math.SQRT2*Math.exp(bias/(2*temp));
    const gap=Math.hypot(a-d,2*b), top=(a+d+gap)/2;
    // Use the determinant for the lower eigenvalue to retain its significant digits.
    const low=(a*d-b*b)/top, odd=Math.expm1(1/temp), second=Math.max(low,odd);
    return {top,second,length:1/Math.log(top/second)};
  }
  model("colour-chain",[["Reduced temperature kBT/J",.4,3,.02,1],["Reduced colour field H/J",-.5,.5,.01,.15]],"Reduced temperature kBT/J","Correlation length (sites)",([t,h])=>{
    const xx=samples(.4,3), selected=chainModes(t,h);
    return {curves:[line("Selected colour bias",xx,v=>chainModes(v,h).length),line("Zero bias",xx,v=>chainModes(v,0).length),point("Selected temperature",t,selected.length)],
      result:"Leading eigenvalue "+display(selected.top)+", second "+display(selected.second)+", longest length "+display(selected.length)+" sites."};
  },local(([t,h])=>"chain --temperature "+t+" --bias "+h));
  model("finite-solids",[["Reservoir modes",2,40,1,8],["Excitation quanta",4,65,1,24]],"Quanta in the four-mode sample","Probability per integer count",([b,qmax])=>{
    const a=4, s=a+b, p=[1], canonical=[], occ=qmax/s;
    let total=1;
    canonical.push((1/(1+occ))**a);
    for(let q=0;q<qmax;q++){
      p.push(p[q]*(q+a)/(q+1)*(qmax-q)/(qmax-q+b-1));total+=p[q+1];
      canonical.push(canonical[q]*(q+a)/(q+1)*occ/(1+occ));
    }
    const xx=Array.from({length:qmax+1},(_,i)=>i);
    return {curves:[line("Isolated pair of solids",xx,q=>p[q]/total),line("Canonical sample",xx,q=>canonical[q])],
      result:"Mean "+display(qmax*a/s)+" quanta; closed-pair variance "+display(qmax*a*b*(s+qmax)/(s*s*(s+1)))+". Canonical variance "+display(a*occ*(1+occ))+"."};
  });
  model("work-histograms",[["Initial gap (kBT)",.5,6,.05,4],["Independent preparations",1,200,1,20]],"Number of independently prepared samples","Probability of missing all excited states",([gap,n])=>{
    const p=1/(1+Math.exp(gap)), xx=Array.from({length:200},(_,i)=>i+1), no=x=>Math.exp(x*Math.log1p(-p));
    return {curves:[line("No initially excited sample",xx,no),line("Five percent",xx,()=>.05),point("Selected experiment size",n,no(n))],
      result:"P(no rare work sample) = "+display(no(n))+". Mean work / kBT = "+display(-p*gap)+", ΔF/kBT = "+display(Math.log1p(Math.exp(-gap))-Math.log(2))+"."};
  });
  model("fastest-descent",[["Horizontal distance X (m)",.2,3,.02,1.58],["Vertical drop Y (m)",.3,1.8,.02,1]],"Horizontal position x (m)","Depth y below release (m)",([x,y])=>{
    let lo=0, hi=2*pi;
    for (let i=0;i<54;i++) {
      const u=(lo+hi)/2;
      if ((u-Math.sin(u))/(1-Math.cos(u))<x/y) lo=u; else hi=u;
    }
    const end=(lo+hi)/2,a=y/(1-Math.cos(end)),xx=samples(0,end);
    return {curves:[{name:"Cycloidal track",points:xx.map(u=>[a*(u-Math.sin(u)),a*(1-Math.cos(u))])},
      {name:"Straight track",points:[[0,0],[x,y]]}],
      result:"Travel time: cycloid "+display(end*Math.sqrt(a/9.81))+" s, straight "+display(Math.sqrt(2*(x*x+y*y)/(9.81*y)))+" s. Final cycloid parameter "+display(end)+" rad."};
  });
  model("vibrated-pendulum",[["Vertical amplitude (mm)",1,9,.1,6],["Drive frequency (Hz)",35,85,1,60]],"Elapsed time (s)","Deviation from upright (degrees)",([amplitude,f])=>{
    const a=amplitude/1000, l=.15, w=2*pi*f, g=9.81, strength=(a*w/l)**2/2;
    const h=1/(f*48), k=(theta,time,exact)=>exact
      ? -(g-a*w*w*Math.cos(w*time))*Math.sin(theta)/l
      : -(g/l+strength*Math.cos(theta))*Math.sin(theta);
    function run(exact) {
      let theta=pi+.02,velocity=0;
      const points=[[0,(theta-pi)*180/pi]], steps=Math.round(1.2*f)*48;
      for(let i=0;i<steps;i++){
        const time=i*h, a1=k(theta,time,exact), v2=velocity+.5*h*a1;
        const a2=k(theta+.5*h*velocity,time+.5*h,exact), v3=velocity+.5*h*a2;
        const a3=k(theta+.5*h*v2,time+.5*h,exact), v4=velocity+h*a3;
        const a4=k(theta+h*v3,time+h,exact);
        theta+=h*(velocity+2*v2+2*v3+v4)/6;
        velocity+=h*(a1+2*a2+2*a3+a4)/6;
        if((i+1)%48===0) points.push([(i+1)*h,(theta-pi)*180/pi]);
      }
      return points;
    }
    const curvature=strength-g/l;
    return {curves:[{name:"Full motion sampled once per drive cycle",points:run(true)},{name:"Averaged motion",points:run(false)}],
      result:"Threshold amplitude "+display(1000*Math.sqrt(2*g*l)/w)+" mm. Averaged upright curvature "+display(curvature)+" s⁻²; "+(curvature>0?"oscillation frequency "+display(Math.sqrt(curvature)/(2*pi))+" Hz.":"growth rate "+display(Math.sqrt(-curvature))+" s⁻¹.")};
  });
  model("spin-tomography",[["Up counts along y (out of 25)",0,25,1,19],["Up counts along z (out of 25)",0,25,1,19]],"Angle from reconstructed spin (degrees)","Predicted spin-up probability",([y,z])=>{
    const radius=Math.hypot(.84,2*y/25-1,2*z/25-1), physical=Math.min(1,radius), xx=samples(0,180);
    return {curves:[line("Linear inversion",xx,a=>(1+radius*Math.cos(a*pi/180))/2),line("Euclidean physical estimate",xx,a=>(1+physical*Math.cos(a*pi/180))/2)],
      result:"Reconstructed Bloch radius "+display(radius)+". Eigenvalues "+display((1+radius)/2)+" and "+display((1-radius)/2)+"."};
  });
  model("two-photon-interference",[["Single-photon rms intensity width σ (fs)",30,160,2,80],["Relative polarisation angle (degrees)",0,90,1,30]],"Relative delay (fs)","Coincidence probability",([width,degrees])=>{
    const xx=samples(-520,520,280), v=Math.cos(degrees*pi/180)**2;
    const prob=(delay,overlap)=>(1-overlap*Math.exp(-delay*delay/(4*width*width)))/2;
    return {curves:[line("Chosen polarisation",xx,t=>prob(t,v)),line("Matched polarisation",xx,t=>prob(t,1)),line("Fully distinguishable",xx,()=>.5)],
      result:"Zero-delay coincidence "+display((1-v)/2)+". "+(v>1e-12?"Full width at half depth "+display(4*width*Math.sqrt(Math.log(2)))+" fs.":"At orthogonal polarisation the dip vanishes and its width cannot be measured.")};
  });
  model("effusive-flight",[["Argon temperature (K)",120,650,5,300],["Flight length (m)",.2,1.5,.02,.5]],"Arrival time (ms)","Arrival density (per ms)",([temperature,length])=>{
    const v=Math.sqrt(2*1.380649e-23*temperature/(39.95*1.6605390666e-27)), t0=1000*length/v;
    const xx=samples(.16*t0,3.5*t0,220), density=(t,beam)=>{
      const s=t/t0;
      return (beam?2/s**5:4/(Math.sqrt(pi)*s**4))*Math.exp(-1/(s*s))/t0;
    };
    return {curves:[line("Aperture flux",xx,t=>density(t,true)),line("Volume-sampled source",xx,t=>density(t,false))],
      result:"Thermal speed "+display(v)+" m/s. Most probable effusive arrival "+display(t0*Math.sqrt(.4))+" ms. Interpreting L/t as a gas speed would give "+display(Math.sqrt(2.5)*v)+" m/s."};
  });
  model("hidden-currents",[["Clockwise bias",-.85,.85,.01,.5],["Total jump rate (s⁻¹)",1,14,.25,8]],"Observation time (s)","Net clockwise jumps",([bias,rate])=>{
    const xx=samples(0,18), mean=t=>rate*bias*t, production=rate*bias*(Math.log1p(bias)-Math.log1p(-bias));
    return {curves:[line("Mean signed count",xx,mean),line("Mean plus one standard deviation",xx,t=>mean(t)+Math.sqrt(rate*t)),
      line("Mean minus one standard deviation",xx,t=>mean(t)-Math.sqrt(rate*t))],
      result:"Every stationary population is 1/3. Net clockwise edge current "+display(rate*bias/3)+" s⁻¹. Entropy production "+display(production)+" kB/s."};
  });
  model("normal-force",[["Bottom speed (m/s)",1,7,.05,4],["Loop radius (m)",.2,1.2,.02,.5]],"Angle from the bottom (degrees)","Inward groove reaction (N)",([v,r])=>{
    const scaled=v*v/(9.81*r), top=scaled>=4?180:Math.acos(1-scaled/2)*180/pi, xx=samples(0,top);
    const force=deg=>.1*9.81*(scaled-2+3*Math.cos(deg*pi/180));
    return {curves:[line("Required radial reaction",xx,force),line("Loss of one-sided support",xx,()=>0)],
      result:"Reachable angle "+display(top)+"°. Inward groove reaction there "+display(force(top))+" N. One-sided loop threshold v₀ = "+display(Math.sqrt(5*9.81*r))+" m/s."};
  });
  model("orbit-inference",[["Initial radial velocity",-.4,.4,.01,.1],["Initial tangential velocity",.55,1.2,.01,.9]],"Azimuth from the initial separation (degrees)","Separation / initial separation",([vr,vt])=>{
    const p=vt*vt, ex=p-1, ey=-vt*vr,e=Math.hypot(ex,ey), xx=samples(0,360,300);
    return {curves:[line("Bound orbit",xx,a=>p/(1+ex*Math.cos(a*pi/180)+ey*Math.sin(a*pi/180))),line("Initial separation",xx,()=>1)],
      result:"Eccentricity "+display(e)+". Closest approach "+display(p/(1+e))+"; largest separation "+display(p/(1-e))+"."};
  });
  model("scattering-bins",[["Polar detector bins",4,24,1,8],["Effective radius (cm)",.5,3,.05,1]],"Bin centre (degrees)","Integrated cross-section (cm²)",([n,r])=>{
    const h=pi/n,area=pi*r*r, exact=[],approx=[];
    for(let i=0;i<n;i++){
      const m=(i+.5)*h,x=m*180/pi;
      exact.push([x,area*(Math.cos(i*h)-Math.cos((i+1)*h))/2]);approx.push([x,area*h*Math.sin(m)/2]);
    }
    return {curves:[{name:"Incoming annulus",points:exact},{name:"Midpoint solid angle",points:approx}],
      result:"Exact sum "+display(area)+" cm². Midpoint sum "+display(approx.reduce((s,p)=>s+p[1],0))+" cm². Differential cross-section "+display(r*r/4)+" cm²/sr."};
  });
  model("hoop-equilibria",[["Hoop radius (m)",.15,.65,.01,.30],["Imposed rotation (rad/s)",2,13,.1,6]],"Angle from the bottom (degrees)","Effective potential U/(mgR)",([r,w])=>{
    const k=w*w*r/9.81,xx=samples(-165,165,260), energy=d=>-Math.cos(d*pi/180)-.5*k*Math.sin(d*pi/180)**2;
    const angle=k>1?Math.acos(1/k)*180/pi:0;
    return {curves:[line("Rotating hoop",xx,energy),line("Stationary hoop",xx,d=>-Math.cos(d*pi/180)),
      {name:"Stable angle(s)",points:angle?[[-angle,energy(-angle)],[angle,energy(angle)]]:[[0,-1]]}],
      result:"Critical speed "+display(Math.sqrt(9.81/r))+" rad/s. Bottom curvature in units of mgR is "+display(1-k)+". Stable angular magnitude "+display(angle)+"°."};
  });
  model("gauge",[["Uniform magnetic field B (T)",-.8,.8,.02,.3],["Gauge coefficient g (T)",-.8,.8,.02,.2]],"Position x on the line y = 0 (m)","Vector potential component Ay (T m)",([b,g])=>{
    const xx=samples(-1,1);
    return {curves:[line("Symmetric gauge",xx,x=>b*x/2),line("With gradient of gxy",xx,x=>(b/2+g)*x)],
      result:"∂xAy = "+display(b/2+g)+" T, ∂yAx = "+display(-b/2+g)+" T. Curl = "+display(b)+" T. Both loop integrals around 0.30 × 0.20 m are "+display(b*.06)+" Wb."};
  });
  model("coherent",[["Coherent amplitude magnitude",0,3,.05,1.3],["Preparation phase (degrees)",-180,180,2,30]],"Time in oscillator periods","Position in units of √(ℏ/mω)",([alpha,phi])=>{
    const xx=samples(0,2,260), c=t=>Math.SQRT2*alpha*Math.cos(2*pi*t-phi*pi/180);
    return {curves:[line("Mean position",xx,c),line("Mean plus one rms width",xx,t=>c(t)+1/Math.SQRT2),line("Mean minus one rms width",xx,t=>c(t)-1/Math.SQRT2)],
      result:"Mean occupation "+display(alpha*alpha)+" quanta, standard deviation "+display(alpha)+". Ground-state probability "+display(Math.exp(-alpha*alpha))+". Position variance in the plotted units is 1/2."};
  });
  model("zeeman",[["Spin–orbit coefficient A (µeV)",16,100,2,48],["Selected magnetic field (T)",0,3,.02,.6]],"Magnetic field (tesla)","Energy at m = +1/2 (µeV)",([a,field])=>{
    const xx=samples(0,3), mu=57.883818, eig=(B,sign)=>{
      const u=mu*B,v=-a/2;
      return (u+v+sign*Math.hypot(u-v,Math.SQRT2*a))/2;
    };
    return {curves:[line("Upper mixed state",xx,B=>eig(B,1)),line("Lower mixed state",xx,B=>eig(B,-1)),line("Weak-field j = 3/2",xx,B=>a/2+2*mu*B/3),line("Weak-field j = 1/2",xx,B=>-a+mu*B/3)],
      result:"Selected energies "+display(eig(field,-1))+" and "+display(eig(field,1))+" µeV. The ratio μB B/A is "+display(mu*field/a)+"."};
  });
  model("exchange",[["On-site energy U (meV)",1,20,.2,6],["Hopping t (meV)",.3,1.6,.05,1]],"Repulsion U/t","Singlet–triplet gap / t",([u,t])=>{
    const xx=samples(.6,22), exact=r=>8/(Math.hypot(r,4)+r), ratio=u/t;
    return {curves:[line("Exact two-site singlet",xx,exact),line("Virtual hopping 4t/U in these units",xx,r=>4/r),point("Selected dimer",ratio,exact(ratio))],
      result:"Triplet energy 0. Lower singlet energy "+display(-t*exact(ratio))+" meV. Estimated gap 4t²/U = "+display(4*t*t/u)+" meV."};
  });
  function airyOnGrid() {
    // Ai(0) and Ai'(0) fix the decaying Airy solution of y'' = z y, including at z = 0.
    const centre=[.35502805388781724,-.2588194037928068], sides=[];
    for (const direction of [-1,1]) {
      const values=[[0,centre[0]]], h=direction*.01, steps=direction<0?600:300;
      let y=centre[0],v=centre[1];
      for(let i=0;i<steps;i++){
        const z=i*h, a1=z*y, v2=v+h*a1/2, a2=(z+h/2)*(y+h*v/2);
        const v3=v+h*a2/2, a3=(z+h/2)*(y+h*v2/2),v4=v+h*a3,a4=(z+h)*(y+h*v3);
        y+=h*(v+2*v2+2*v3+v4)/6;v+=h*(a1+2*a2+2*a3+a4)/6;
        values.push([(i+1)*h,y]);
      }
      sides.push(values);
    }
    return sides[0].reverse().concat(sides[1].slice(1));
  }
  const airyValues=airyOnGrid();
  model("turning-point",[["Turning-point length (nm)",.3,1.8,.02,.8],["Turning-point position (nm)",-2,2,.05,.4]],"Laboratory coordinate x (nm)","Wave amplitude (common normalisation)",([length,x0])=>{
    const allowed=samples(-6,-.35,200), forbidden=samples(.35,3);
    const at=(name,zz,fn)=>({name,points:zz.map(z=>[x0+length*z,fn(z)])});
    return {curves:[{name:"Numerically integrated Ai",points:airyValues.map(([z,a])=>[x0+length*z,a])},
      at("Allowed WKB",allowed,z=>Math.sin(2*(-z)**1.5/3+pi/4)/(Math.sqrt(pi)*(-z)**.25)),
      at("Forbidden WKB",forbidden,z=>Math.exp(-2*z**1.5/3)/(2*Math.sqrt(pi)*z**.25))],
      result:"The turn at "+display(x0)+" nm has finite Ai value 0.35503. The characteristic transition length is "+display(length)+" nm; the asymptotic curves omit the nearest 0.35 lengths."};
  });
  model("mixing",[["Mole fraction of component A",0,1,.01,.3],["Separation reservoir temperature (K)",200,500,5,300]],"Mole fraction of A","Ideal molar mixing entropy (J mol⁻¹ K⁻¹)",([x,t])=>{
    const xx=samples(0,1), term=a=>a===0?0:-a*Math.log(a), s=a=>8.314462618*(term(a)+term(1-a));
    return {curves:[line("Two distinguishable ideal components",xx,s),line("Identical gas on both sides",xx,()=>0),point("Selected composition",x,s(x))],
      result:"Ideal entropy increase "+display(s(x))+" J mol⁻¹ K⁻¹. Reversible isothermal separation of the distinguishable mixture requires "+display(t*s(x)/1000)+" kJ/mol."};
  });
  model("rotational-heat",[["Rotational temperature θrot (K)",2,20,.2,8],["Chosen gas temperature (K)",1,95,1,20]],"Gas temperature (K)","Rotational heat capacity per molecule / kB",([theta,t])=>{
    const heat=T=>{
      let z=0,m=0,m2=0;
      for(let j=0;j<=70;j++){
        const dimensionless=theta*j*(j+1)/T, weight=(2*j+1)*Math.exp(-dimensionless);
        z+=weight;m+=weight*dimensionless;m2+=weight*dimensionless**2;
      }
      return m2/z-(m/z)**2;
    };
    const xx=samples(1,95,180);
    return {curves:[line("Heteronuclear rigid rotor",xx,heat),line("Two quadratic rotations",xx,()=>1),point("Selected temperature",t,heat(t))],
      result:"First excited rotational level has E₁/kB = "+display(2*theta)+" K. At the selected temperature Crot/kB = "+display(heat(t))+". Vibrational and translational contributions are excluded."};
  });
  model("kinetic-collisions",[["Initial fraction in velocity pair A",.01,.99,.01,.83],["Collision rate coefficient (s⁻¹)",.2,3,.05,.8]],"Time (s)","Shannon entropy per particle / kB",([x0,k])=>{
    const xx=samples(0,6), population=t=>.5+(x0-.5)*Math.exp(-2*k*t), entropy=x=>-x*Math.log(x/2)-(1-x)*Math.log((1-x)/2);
    return {curves:[line("Four-state collision ensemble",xx,t=>entropy(population(t))),line("Equal populations",xx,()=>Math.log(4))],
      result:"Initial clockwise-minus-reverse pair imbalance x²−(1−x)² = "+display(2*x0-1)+". Initial d(S/kB)/dt = "+display(-k*(2*x0-1)*Math.log((1-x0)/x0))+" s⁻¹."};
  });
  model("green-kubo",[["Correlation decay time τ (ps)",.3,3,.05,1],["Oscillation frequency Ω (rad/ps)",0,4,.05,1]],"Upper integration time (ps)","Running diffusivity (nm²/ps)",([tau,w])=>{
    const rate=1/tau, exact=rate/(rate*rate+w*w), xx=samples(0,12,260);
    const integral=t=>(rate+Math.exp(-rate*t)*(-rate*Math.cos(w*t)+w*Math.sin(w*t)))/(rate*rate+w*w);
    return {curves:[line("Finite Green–Kubo integral",xx,integral),line("Infinite-time transport coefficient",xx,()=>exact)],
      result:"With C(0) = 1 nm²/ps² the exact 1D diffusion coefficient is "+display(exact)+" nm²/ps. At 2τ the integral gives "+display(integral(2*tau))+" nm²/ps."};
  });
  model("nucleation",[["Interfacial energy (mJ/m²)",10,90,1,35],["Bulk driving density (MJ/m³)",5,80,1,25]],"Cluster radius (nm)","Free energy (eV)",([gammaMilli,driveMega])=>{
    const gamma=gammaMilli*.001, drive=driveMega*1e6, rstar=2*gamma/drive, ev=1.602176634e-19;
    const surface=nm=>4*pi*gamma*(nm*1e-9)**2/ev, bulk=nm=>-4*pi*drive*(nm*1e-9)**3/(3*ev);
    const xx=samples(0,1.85*rstar*1e9), top=surface(rstar*1e9)+bulk(rstar*1e9);
    return {curves:[line("Total free energy",xx,r=>surface(r)+bulk(r)),line("Surface cost",xx,surface),line("Bulk gain",xx,bulk),point("Critical cluster",rstar*1e9,top)],
      result:"Critical radius "+display(rstar*1e9)+" nm; barrier "+display(top)+" eV. The free-energy curve crosses zero again at "+display(1.5*rstar*1e9)+" nm."};
  });
})();
