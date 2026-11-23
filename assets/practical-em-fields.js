(() => {
  const fmt = x => Number(x.toPrecision(4)).toString();
  const pi = Math.PI;
  function curve(name, from, to, intervals, f) {
    return {name, points:Array.from({length:intervals+1}, (_,i) => {
      const x=from+(to-from)*i/intervals;
      return [x,f(x)];
    })};
  }
  function probe(name,x,y) { return {name,points:[[x,y]]}; }

  // Evaluate the Gaussian tail on the positive half-line using composite Simpson quadrature.
  function erfc(x) {
    if (x === 0) return 1;
    if (x >= 9) return 0;
    const intervals=120, h=(9-x)/intervals;
    let sum=Math.exp(-x*x)+Math.exp(-81);
    for(let j=1;j<intervals;j++) sum+=(j%2===0?2:4)*Math.exp(-((x+j*h)**2));
    return 2*h*sum/(3*Math.sqrt(pi));
  }

  Object.assign(window.fundamentalStudies ||= {}, {
    'em-coefficient-network': {
      controls:[['Mutual capacitance (pF)',0,12,0.25,4],['Drive voltage V₁ (V)',-3,3,0.1,2]],
      xLabel:'Drive voltage V₁ (V)', yLabel:'Input electrode charge (pC)',
      calculate([mutual,drive]) {
        const g1=2, g2=3, vg=mutual*drive/(g2+mutual);
        const grounded=g1+mutual, floating=g1+mutual*g2/(mutual+g2);
        return {curves:[curve('Second electrode grounded',-3,3,60,x=>grounded*x),curve('Second electrode neutral and floating',-3,3,60,x=>floating*x),
          probe('Selected floating charge on electrode 1',drive,floating*drive)],
          result:'Neutral electrode 2 floats to '+fmt(vg)+' V. Grounded input capacitance '+fmt(grounded)+' pF, floating input capacitance '+fmt(floating)+' pF. Floating-case field energy '+fmt(0.5*floating*drive**2)+' pJ.'};
      }
    },
    'em-magnetic-diffusion': {
      controls:[['Time after field step (ms)',0.025,0.5,0.025,0.1],['Sinusoidal drive frequency (kHz)',0.25,8,0.25,1],['Probe depth (mm)',0,7,0.1,2]],
      xLabel:'Depth in copper (mm)',yLabel:'Field / applied amplitude',
      calculate([ms,kHz,depth]) {
        const diffusivity=1/(4*pi*1e-7*5.8e7);
        const root=Math.sqrt(diffusivity*ms*1e-3)*1e3;
        const skin=Math.sqrt(diffusivity/(pi*kHz*1e3))*1e3;
        const step=x=>erfc(x/(2*root)), envelope=x=>Math.exp(-x/skin);
        return {curves:[curve('Field after a step',0,7,140,step),curve('Sinusoidal amplitude envelope',0,7,140,envelope),probe('Step at probe',depth,step(depth)),probe('AC amplitude at probe',depth,envelope(depth))],
          result:'Half-step depth '+fmt(0.9538725524*root)+' mm. AC skin depth '+fmt(skin)+' mm. At the probe the step fraction is '+fmt(step(depth))+', AC amplitude fraction '+fmt(envelope(depth))+' and AC phase lag '+fmt(depth/skin)+' rad.'};
      }
    },
    'em-plasma-screening': {
      controls:[['Electron density (10¹⁶ m⁻³)',0.25,5,0.25,1],['Electron thermal energy (eV)',0.25,8,0.25,2],['Probe radius (µm)',5,800,5,105]],
      xLabel:'Radius from weak probe (µm)',yLabel:'Ratio to unscreened value',
      calculate([density,thermal,radius]) {
        const n=density*1e16, e=1.602176634e-19, eps=8.8541878128e-12, m=9.1093837e-31;
        const lambda=Math.sqrt(eps*thermal/(n*e)), microns=1e6*lambda, omega=Math.sqrt(n*e*e/(eps*m));
        const potential=r=>Math.exp(-r/microns), field=r=>(1+r/microns)*potential(r);
        return {curves:[curve('Potential ratio',0,800,180,potential),curve('Radial field ratio',0,800,180,field),probe('Potential at probe',radius,potential(radius)),probe('Field at probe',radius,field(radius))],
          result:'Debye length '+fmt(microns)+' µm, plasma frequency '+fmt(omega/(2*pi*1e6))+' MHz, period '+fmt(2*pi*1e9/omega)+' ns. The Debye sphere contains approximately '+fmt(4*pi*n*lambda**3/3)+' electrons.'};
      }
    },
    'em-stokes-analyser': {
      controls:[['Polarised fraction P',0,1,0.05,0.6],['Azimuth ψ (degrees)',-90,90,1,20],['Ellipticity χ (degrees)',-45,45,1,10]],
      xLabel:'Linear analyser angle (degrees)',yLabel:'Transmitted intensity / total',
      calculate([p,azimuth,ellipticity]) {
        const doubleAngle=angle=>2*pi*angle/180;
        const a=doubleAngle(azimuth), b=doubleAngle(ellipticity);
        const q=p*Math.cos(b)*Math.cos(a), u=p*Math.cos(b)*Math.sin(a), v=p*Math.sin(b);
        const scan=(angle,cross)=>(1+q*Math.cos(doubleAngle(angle))+cross*Math.sin(doubleAngle(angle)))/2;
        const pairs=[[scan(0,u),scan(90,u)],[scan(45,u),scan(135,u)],[scan(45,v),scan(135,v)]];
        const reconstructed=pairs.map(([positive,negative])=>positive-negative);
        return {curves:[curve('Linear analyser alone',0,180,180,x=>scan(x,u)),curve('Quarter-wave plate before analyser',0,180,180,x=>scan(x,v))],
          result:'Measured complementary pairs (horizontal, diagonal, circular): '+pairs.map(pair=>pair.map(fmt).join(' / ')).join(' · ')+'. Reconstructed (S₁,S₂,S₃)/S₀ = ('+reconstructed.map(fmt).join(', ')+'); degree '+fmt(Math.hypot(...reconstructed))+'.'};
      }
    },
    'em-helmholtz': {
      controls:[['Cosine amplitude aₓ',-4,4,0.25,3],['Cosine amplitude aᵧ',-4,4,0.25,1],['Uniform component bₓ',-2,2,0.25,1]],
      xLabel:'Phase (2x + y) / π',yLabel:'x-component of field',
      calculate([ax,ay,bias]) {
        const projection=(2*ax+ay)/5, lx=2*projection, ly=projection, tx=ax-lx, ty=ay-ly;
        const longPower=lx*lx+ly*ly, transPower=tx*tx+ty*ty;
        return {curves:[curve('Measured x-component',0,2,160,x=>bias+ax*Math.cos(pi*x)),curve('Longitudinal x-component',0,2,160,x=>lx*Math.cos(pi*x)),curve('Transverse x-component',0,2,160,x=>tx*Math.cos(pi*x)),curve('Uniform x-component',0,2,1,()=>bias)],
          result:'Longitudinal vector ('+fmt(lx)+', '+fmt(ly)+'), transverse ('+fmt(tx)+', '+fmt(ty)+'). Squared amplitudes add to '+fmt(longPower)+' + '+fmt(transPower)+' = '+fmt(ax*ax+ay*ay)+'. The constant x-component is '+fmt(bias)+'.'};
      }
    }
  });
})();
