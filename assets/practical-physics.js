// Interactive calculations for the physics book chapters. All axes use the units named in their labels.
(() => {
  const pi = Math.PI;
  const radians = degrees => degrees * pi / 180;
  const format = value => Number(value.toPrecision(4)).toString();

  function samples(name, first, last, intervals, fn) {
    const points = Array.from({length: intervals + 1}, (_, i) => {
      const x = first + (last - first) * i / intervals;
      return [x, fn(x)];
    });
    return {name, points};
  }

  function marker(name, x, y) {
    return {name, points: [[x, y]]};
  }

  Object.assign(window.fundamentalStudies ||= {}, {
    'ph-moving-trap': {
      controls: [['Support acceleration (m/s²)', -3, 3, 0.1, 1.6], ['Spring stiffness (N/m)', 2, 16, 0.5, 8]],
      xLabel: 'Time (s)', yLabel: 'Position relative to support (m)',
      calculate([acceleration, stiffness]) {
        const mass = 0.5, omega = Math.sqrt(stiffness / mass);
        const equilibrium = -mass * acceleration / stiffness;
        return {curves: [
          samples('Bead', 0, 4, 180, t => equilibrium * (1 - Math.cos(omega * t))),
          samples('Accelerated equilibrium', 0, 4, 1, () => equilibrium),
        ], result: acceleration === 0 ? 'The equilibrium stays at the origin and the bead remains at rest.' :
          'The equilibrium is at ' + format(equilibrium) + ' m. The first turning point is ' + format(2 * equilibrium) + ' m at ' + format(pi / omega) + ' s.'};
      }
    },
    'ph-action-clock': {
      controls: [['Upward launch speed (m/s)', 1, 7, 0.1, 4], ['Gate height (m)', 0.1, 1.7, 0.05, 0.5]],
      xLabel: 'Time after launch (s)', yLabel: 'Height (m)',
      calculate([speed, height]) {
        const gravity = 9.81, discriminant = speed ** 2 - 2 * gravity * height;
        const flight = v => samples(v === speed ? 'Selected launch' : 'Reference launch', 0, 2*v/gravity, 120, t => t*(v-gravity*t/2));
        const curves = [flight(speed), samples('Observation gate', 0, 2*Math.max(speed,4)/gravity, 1, () => height)];
        if (speed !== 4) curves.push(flight(4));
        let result = 'Peak height ' + format(speed**2/(2*gravity)) + ' m. ';
        if (discriminant < 0) result += 'The trajectory stays below the gate.';
        else {
          const offset = Math.sqrt(discriminant);
          const early = (speed - offset) / gravity, late = (speed + offset) / gravity;
          result += 'Ascending crossing at ' + format(early) + ' s and descending crossing at ' + format(late) + ' s.';
          curves.push({name:'Gate crossings', points:[[early,height],[late,height]]});
        }
        return {curves, result};
      }
    },
    'ph-v-action': {
      controls: [['Final to initial force ratio', 0.25, 8, 0.25, 2]],
      xLabel: 'Position (m)', yLabel: 'Momentum (kg m/s)',
      calculate([ratio]) {
        const orbit = (name, r) => {
          const force = 2*r, energy = 0.16 * r**(2/3), amplitude = energy/force;
          const momentum = q => Math.sqrt(Math.max(0, 0.5*(energy - force*Math.abs(q))));
          const upper = samples(name, -amplitude, amplitude, 100, momentum);
          upper.points.push(...samples(name, amplitude, -amplitude, 100, q => -momentum(q)).points);
          return upper;
        };
        const amplitude = 0.08*ratio**(-1/3);
        return {curves: [orbit('Initial orbit',1), orbit('Slowly adjusted orbit',ratio)],
          result: 'Final amplitude ' + format(amplitude) + ' m, energy ' + format(2*ratio*amplitude) + ' J and period ' + format(4*Math.sqrt(0.5*amplitude/(2*ratio))) + ' s.'};
      }
    },
    'ph-rod-chain': {
      controls: [['Sites per wavelength', 2, 40, 1, 10]],
      xLabel: 'Sites per wavelength', yLabel: 'Speed / continuum speed',
      calculate([n]) {
        const phase = x => Math.sin(pi/x)/(pi/x), group = x => Math.cos(pi/x);
        return {curves:[samples('Phase speed',2,40,160,phase), samples('Group speed',2,40,160,group),
          marker('Selected phase speed',n,phase(n)), marker('Selected group speed',n,group(n))],
          result: 'Frequency and phase speed are ' + format(100*phase(n)) + '% of the continuum prediction. Group speed is ' + format(100*Math.max(0, n === 2 ? 0 : group(n))) + '%.'};
      }
    },
    'ph-proper-clock': {
      controls: [['Speed / speed of light', 0.1, 0.99, 0.01, 0.8], ['Gate separation (light-µs)', 2, 30, 1, 12]],
      xLabel:'Gate separation (light-µs)', yLabel:'Crossing interval (µs)',
      calculate([beta, distance]) {
        const lab = x => x / beta, clock = x => lab(x)*Math.sqrt(1-beta**2);
        return {curves:[samples('Laboratory clocks',0,30,60,lab),samples('Travelling clock',0,30,60,clock),
          marker('Selected laboratory time',distance,lab(distance)),marker('Selected proper time',distance,clock(distance))],
          result:'The laboratory records ' + format(lab(distance)) + ' µs. The travelling clock records ' + format(clock(distance)) + ' µs.'};
      }
    },
    'ph-chirped-packet': {
      controls:[['Initial phase curvature b',-4,4,0.1,-2]],
      xLabel:'Scaled time u', yLabel:'Position standard deviation (µm)',
      calculate([b]) {
        const width = u => 4*Math.hypot(1+b*u,u), focus = Math.max(0,-b/(1+b*b));
        return {curves:[samples('Selected preparation',0,1.5,150,width),samples('Unchirped packet',0,1.5,150,u=>4*Math.hypot(1,u)),marker('Narrowest future width',focus,width(focus))],
          result:'For u ≥ 0 the minimum width is ' + format(width(focus)) + ' µm at u = ' + format(focus) + '.'};
      }
    },
    'ph-ring-flux': {
      controls:[['Signed flux qΦ/h',-1.5,1.5,0.05,0.3]], xLabel:'Signed flux qΦ/h', yLabel:'Energy / Eₗ',
      calculate([flux]) {
        const states = [-2,-1,0,1,2];
        const curves = states.map(n=>samples('n = '+n,-1.5,1.5,120,x=>(n-x)**2));
        const minimum = Math.min(...states.map(n=>(n-flux)**2));
        const ground = states.filter(n=>Math.abs((n-flux)**2-minimum)<1e-9);
        curves.push(marker('Selected ground energy',flux,minimum));
        return {curves, result:'Lowest n: '+ground.join(' and ')+'. Ground energy '+format(minimum)+' Eₗ. The n = 0 current in units of qh/(mL²) is '+format(-flux)+'.'};
      }
    },
    'ph-coulomb-radii': {
      controls:[['Nuclear charge Z',1,5,1,1]], xLabel:'Radius in original Bohr radii', yLabel:'Radial probability per original a₀',
      calculate([z]) {
        const densityS = x => z*(z*x)**2*(2-z*x)**2*Math.exp(-z*x)/8;
        const densityP = x => z*(z*x)**4*Math.exp(-z*x)/24;
        return {curves:[samples('2s radial density',0,16,320,densityS),samples('2p radial density',0,16,320,densityP),marker('2s interior node',2/z,0)],
          result:'The 2s node is at '+format(2/z)+' original Bohr radii. The most probable 2p radius is '+format(4/z)+' original Bohr radii.'};
      }
    },
    'ph-scattering-well': {
      controls:[['Well phase KR',0.1,3.6,0.01,0.8], ['Incoming wave number kR',0.03,0.25,0.01,0.10]], xLabel:'Well phase KR', yLabel:'s-wave cross section / R²',
      calculate([selected,k]) {
        const cross = x => {
          const denominator = x * Math.cos(x), numerator = denominator - Math.sin(x);
          return 4*pi*numerator**2/(denominator**2+(k*numerator)**2);
        };
        const peak = pi/2;
        const curves = [samples('Leading low-energy cross section',0.1,3.6,600,cross),marker('Selected depth',selected,cross(selected)),marker('First threshold',peak,cross(peak))];
        return {curves, result:'At KR = '+format(selected)+' the cross section is '+format(cross(selected))+' R². The finite-k limit at resonance is '+format(4*pi/k**2)+' R².'};
      }
    },
    'ph-spin-loop': {
      controls:[['Cone angle (degrees)',0,180,1,60],['Reference phase (degrees)',-180,180,5,0]],
      xLabel:'Cone angle (degrees)', yLabel:'Bright-port probability',
      calculate([theta,chi]) {
        const phase = angle => -pi*(1-Math.cos(radians(angle)));
        const probability = (angle,direction) => (1+Math.cos(radians(chi)+direction*phase(angle)))/2;
        return {curves:[samples('Forward traversal',0,180,180,x=>probability(x,1)),samples('Reverse traversal',0,180,180,x=>probability(x,-1)),marker('Selected forward probability',theta,probability(theta,1))],
          result:'Forward geometric phase '+format(phase(theta)/pi)+'π radians. Forward port probability '+format(probability(theta,1))+', reverse '+format(probability(theta,-1))+'.'};
      }
    },
    'ph-reservoir': {
      controls:[['Sample capacity Cₐ/kᵦ',20,150,5,60],['Plate capacity Cᵦ/kᵦ',10,250,5,60]],
      xLabel:'Energy deviation / (kᵦT)',yLabel:'Probability density',
      calculate([sample,plate]) {
        const variance = sample*plate/(sample+plate), limit = 4*Math.sqrt(sample);
        const gaussian = (x,v) => Math.exp(-x*x/(2*v))/Math.sqrt(2*pi*v);
        return {curves:[samples('Finite reservoir',-limit,limit,180,x=>gaussian(x,variance)),samples('Canonical limit',-limit,limit,180,x=>gaussian(x,sample))],
          result:'Standard deviations are '+format(Math.sqrt(variance))+' kᵦT for the finite plate and '+format(Math.sqrt(sample))+' kᵦT for the infinite bath.'};
      }
    },
    'ph-adsorption': {
      controls:[['Binding energy ε/(kᵦT)',0,6,0.1,2],['Chemical potential μ/(kᵦT)',-8,3,0.1,-2]],
      xLabel:'Chemical potential μ/(kᵦT)',yLabel:'Coverage or variance per site',
      calculate([binding,chemical]) {
        const occupancy = mu => 1/(1+Math.exp(-mu-binding));
        const noise = mu => occupancy(mu)*(1-occupancy(mu));
        return {curves:[samples('Mean coverage',-8,3,165,occupancy),samples('Number variance per site',-8,3,165,noise),marker('Selected coverage',chemical,occupancy(chemical))],
          result:'At this chemical potential, 100 sites have mean occupancy '+format(100*occupancy(chemical))+' particles and standard deviation '+format(Math.sqrt(100*noise(chemical)))+' particles.'};
      }
    },
    'ph-virial': {
      controls:[['Outer radius / core diameter',1.1,1.8,0.01,1.5],['Reduced temperature kᵦT/ε',0.7,8,0.1,2]],
      xLabel:'Reduced temperature kᵦT/ε',yLabel:'Second virial coefficient / Bcore',
      calculate([range,temp]) {
        const shell = range**3-1, virial = t => 1-shell*Math.expm1(1/t);
        const boyle = 1/Math.log1p(1/shell);
        return {curves:[samples('Hard core plus attraction',0.7,8,180,virial),samples('Zero density slope',0.7,8,1,()=>0),marker('Selected temperature',temp,virial(temp))],
          result:'B₂/Bcore = '+format(virial(temp))+'. The Boyle temperature is '+format(boyle)+' ε/kᵦ.'};
      }
    },
    'ph-common-tangent': {
      controls:[['Composition half-gap d',0.05,0.45,0.01,0.3], ['Mean B fraction',0,1,0.01,0.45]],
      xLabel:'Fraction of species B',yLabel:'Tilt-subtracted energy / A',
      calculate([gap,mean]) {
        const left = 0.5-gap, right=0.5+gap;
        const uniform = c=>((c-0.5)**2-gap**2)**2;
        const stable = c=>c>=left && c<=right ? 0 : uniform(c);
        const result=mean>left && mean<right ? 'The B-rich phase occupies '+format(100*(mean-left)/(right-left))+'% of sites. Energy saving is '+format(uniform(mean))+' A per site.' : 'A homogeneous phase has the minimum free energy at the selected composition.';
        return {curves:[samples('Uniform material',0,1,200,uniform),samples('Equilibrium envelope',0,1,200,stable),marker('Selected equilibrium',mean,stable(mean))],result};
      }
    },
    'ph-ising-ring': {
      controls:[['Temperature kᵦT/J',0.3,3,0.05,1], ['Sites around the ring',8,100,2,32]],
      xLabel:'Separation in bonds',yLabel:'Spin correlation',
      calculate([temperature,sites]) {
        const q=Math.tanh(1/temperature);
        const finite=r=>(q**r+q**(sites-r))/(1+q**sites);
        return {curves:[samples('Closed finite ring',0,sites,sites,finite),samples('Infinite chain',0,sites,sites,r=>q**r)],
          result:'Infinite-chain correlation length '+format(-1/Math.log(q))+' lattice spacings. The opposite site on the '+sites+'-site ring has correlation '+format(finite(sites/2))+'.'};
      }
    },
    'ph-interface': {
      controls:[['Glass refractive index',1.2,1.9,0.01,1.5],['Angle of incidence (degrees)',0,85,0.5,50]],
      xLabel:'Incidence angle from normal (degrees)',yLabel:'Reflected intensity fraction',
      calculate([index,angle]) {
        const reflect = (degrees,pol) => {
          const theta = radians(degrees);
          const transmittedCosSquared = 1-(index*Math.sin(theta))**2;
          if (transmittedCosSquared<=0) return 1;
          const tangentialFactor = Math.sqrt(transmittedCosSquared), cosI = Math.cos(theta);
          const a = pol==='s' ? index*cosI : cosI, b = pol==='s' ? tangentialFactor : index*tangentialFactor;
          return ((a-b)/(a+b))**2;
        };
        const critical = Math.asin(1/index)*180/pi, brewster = Math.atan(1/index)*180/pi;
        const tail = angle>critical ? ' Field penetration at 500 nm is '+format(500/(2*pi*Math.sqrt((index*Math.sin(radians(angle)))**2-1)))+' nm.' : ' The transmitted field propagates at the selected angle.';
        return {curves:[samples('s polarisation',0,88,352,x=>reflect(x,'s')),samples('p polarisation',0,88,352,x=>reflect(x,'p')),marker('Selected s',angle,reflect(angle,'s')),marker('Selected p',angle,reflect(angle,'p'))],
          result:'Brewster angle '+format(brewster)+'°, critical angle '+format(critical)+'°.'+tail};
      }
    },
    'ph-slab': {
      controls:[['Normalised half-width V',0.3,5,0.05,1]], xLabel:'Transverse coordinate x/a',yLabel:'Fundamental TE field / centre field',
      calculate([v]) {
        function mode(strength) {
          let lo=0, hi=Math.min(strength,pi/2);
          for(let step=0;step<55;step++) {
            const u=(lo+hi)/2;
            if(u*Math.tan(u)>Math.sqrt(Math.max(0,strength**2-u**2))) hi=u;
            else lo=u;
          }
          const u=(lo+hi)/2, w=Math.sqrt(Math.max(0,strength**2-u**2));
          const field=x=>Math.abs(x)<=1 ? Math.cos(u*x) : Math.cos(u)*Math.exp(-w*(Math.abs(x)-1));
          return {u,w,field};
        }
        const chosen=mode(v);
        return {curves:[samples('Selected fundamental mode',-6,6,240,chosen.field),samples('Reference V = 2',-6,6,240,mode(2).field)],
          result:'Root u = '+format(chosen.u)+', cladding exponent w = '+format(chosen.w)+'. Field amplitude decays by 1/e over '+format(1/chosen.w)+' half-widths outside the core.'};
      }
    },
    'ph-fresnel-zones': {
      controls:[['Aperture radius (mm)',0.05,1.25,0.025,0.5],['Screen distance (m)',0.25,1,0.05,0.5]],
      xLabel:'Aperture radius (mm)', yLabel:'On-axis intensity / free intensity',
      calculate([radius,distance]) {
        const zones=mm=>mm**2/(0.5*distance);
        const intensity=mm=>4*Math.sin(pi*zones(mm)/2)**2;
        return {curves:[samples('Fresnel integral at 500 nm',0,1.25,350,intensity),marker('Selected aperture',radius,intensity(radius))],
          result:'The selected opening spans '+format(zones(radius))+' half-period zones and gives normalised intensity '+format(intensity(radius))+'. First complete zone radius '+format(Math.sqrt(0.5*distance))+' mm.'};
      }
    },
    'ph-cavity': {
      controls:[['Mirror separation / curvature radius',0.1,1.9,0.05,1],['Wavelength (nm)',400,800,10,600]],
      xLabel:'Distance from waist (m)',yLabel:'Beam radius w (mm)',
      calculate([ratio,wavelength]) {
        const radius=0.5, length=ratio*radius, zR=Math.sqrt(length/2*(radius-length/2));
        const waist=1e3*Math.sqrt(wavelength*1e-9*zR/pi), width=z=>waist*Math.hypot(1,z/zR);
        return {curves:[samples('Gaussian beam radius',-length/2,length/2,160,width),marker('Waist',0,waist)],
          result:'Rayleigh length '+format(zR)+' m, waist radius '+format(waist)+' mm, mirror spot radius '+format(width(length/2))+' mm.'};
      }
    },
    'ph-laser': {
      controls:[['Output mirror reflectance',0.70,0.99,0.01,0.90],['Small-signal gain (1/m)',0,1.5,0.05,0.5]],
      xLabel:'Small-signal intensity gain (1/m)',yLabel:'Circulating intensity / saturation intensity',
      calculate([reflectance,pump]) {
        const threshold=0.10-Math.log(0.98*reflectance)/0.60;
        const intensity=g=>Math.max(0,g/threshold-1), lifetime=1e9/(299792458*threshold);
        return {curves:[samples('Single-mode saturated gain',0,1.5,180,intensity),marker('Selected pump',pump,intensity(pump))],
          result:'Threshold '+format(threshold)+' m⁻¹, selected circulating intensity '+format(intensity(pump))+' Iₛ, passive intensity lifetime '+format(lifetime)+' ns.'};
      }
    },
    'ph-dimer-band': {
      controls:[['Intracell hopping t₁ (eV)',0.2,1.8,0.05,1],['Intercell hopping t₂ (eV)',0.2,1.8,0.05,0.6]],
      xLabel:'Wave number ka / π',yLabel:'Band energy (eV)',
      calculate([t1,t2]) {
        const energy=x=>Math.hypot(t1+t2*Math.cos(pi*x),t2*Math.sin(pi*x));
        return {curves:[samples('Conduction band',-1,1,180,energy),samples('Valence band',-1,1,180,x=>-energy(x))],
          result:'Direct gap '+format(2*Math.abs(t1-t2))+' eV. Each band spans '+format(2*Math.min(t1,t2))+' eV.'};
      }
    },
    'ph-end-mode': {
      controls:[['Hopping ratio t₁/t₂',0.25,2.5,0.05,0.5],['Number of A sites',4,30,1,12]],
      xLabel:'Sublattice-site number j',yLabel:'Local probability',
      calculate([ratio,sites]) {
        const weights=Array.from({length:sites},(_,j)=>ratio**(2*j)), norm=weights.reduce((a,b)=>a+b,0);
        const density=j=>weights[j-1]/norm;
        return {curves:[samples('A-sublattice probability',1,sites,sites-1,density),samples('B sublattice',1,sites-1,sites-2,()=>0)],
          result:'Probabilities at the first and last A sites are '+format(density(1))+' and '+format(density(sites))+'. '+(Math.abs(ratio-1)<1e-8?'The zero mode is uniform on A.':'The amplitude decay length from the favoured edge is '+format(1/Math.abs(Math.log(ratio)))+' cell spacings.')};
      }
    },
    'ph-honeycomb': {
      controls:[['Sublattice energy offset Δ (eV)',0,0.5,0.01,0.1],['Selected displacement qₓd',-0.8,0.8,0.01,0.05]],
      xLabel:'Displacement qₓd from K',yLabel:'Conduction energy (eV)',
      calculate([delta,q]) {
        const t=2.7, exact=x=>Math.hypot(delta,t*(1+2*Math.cos(2*pi/3+Math.sqrt(3)*x/2)));
        const cone=x=>Math.hypot(delta,1.5*t*x);
        return {curves:[samples('Honeycomb hopping model',-0.8,0.8,240,exact),samples('Local Dirac model',-0.8,0.8,240,cone),marker('Selected lattice energy',q,exact(q))],
          result:'Direct gap '+format(2*delta)+' eV. At the selected displacement the lattice gives '+format(exact(q))+' eV and the local model '+format(cone(q))+' eV.'};
      }
    },
    'ph-dirac-orbit': {
      controls:[['Signed density (10¹² cm⁻²)',-4,4,0.1,1],['Cone velocity (10⁶ m/s)',0.6,1.4,0.05,1]],
      xLabel:'Signed density (10¹² cm⁻²)',yLabel:'Fermi energy relative to neutrality (eV)',
      calculate([density,speed]) {
        const kF=n=>Math.sqrt(pi*Math.abs(n)*1e16), energy=n=>Math.sign(n)*6.582119569e-16*speed*1e6*kF(n);
        const mass=1.054571817e-34*kF(density)/(speed*1e6*9.1093837e-31);
        return {curves:[samples('Gate-dependent Fermi energy',-4,4,240,energy),marker('Selected filling',density,energy(density))],
          result:density===0?'At neutrality kF and the ideal density of states vanish. There is no finite Fermi orbit.':
            'Fermi energy '+format(energy(density))+' eV, wave number '+format(kF(density)/1e9)+' nm⁻¹, cyclotron mass '+format(mass)+' electron masses.'};
      }
    },
    'ph-dirac-landau': {
      controls:[['Perpendicular field (T)',0.1,10,0.1,1],['Cone velocity (10⁶ m/s)',0.6,1.4,0.05,1]],
      xLabel:'Magnetic field (T)',yLabel:'Landau energy (meV)',
      calculate([field,speed]) {
        const charge=1.602176634e-19, hbar=1.054571817e-34;
        const energy=(b,n)=>1e3/charge*(speed*1e6)*Math.sqrt(2*charge*hbar*b*n);
        const curves=[0,1,2,3].map(n=>samples('Level n = '+n,0,10,240,b=>energy(b,n)));
        curves.push(marker('Selected first level',field,energy(field,1)));
        return {curves,result:'First level '+format(energy(field,1))+' meV, first-to-second spacing '+format(energy(field,2)-energy(field,1))+' meV. Each spin and valley has '+format(charge*field/(2*pi*hbar)*1e-14)+' × 10¹⁰ orbit centres per cm².'};
      }
    }
  });
})();

(() => {
  const pi=Math.PI, fmt=x=>Number(x.toPrecision(4)).toString();
  function sampled(name,start,end,intervals,fn) {
    return {name, points:Array.from({length:intervals+1},(_,j)=>{
      const x=start+(end-start)*j/intervals;
      return [x,fn(x)];
    })};
  }
  function mark(name,x,y) { return {name,points:[[x,y]]}; }

  function floquetExponent(modulation,damping,drive) {
    const omega=10, period=2*pi/drive, steps=64, h=period/steps;
    function advance(position,velocity) {
      const force=(t,q,p)=>-2*damping*p-omega*omega*(1+modulation*Math.cos(drive*t))*q;
      for(let j=0;j<steps;j++) {
        const t=j*h, a=force(t,position,velocity);
        const p2=velocity+h*a/2, a2=force(t+h/2,position+h*velocity/2,p2);
        const p3=velocity+h*a2/2, a3=force(t+h/2,position+h*p2/2,p3);
        const p4=velocity+h*a3, a4=force(t+h,position+h*p3,p4);
        position+=h*(velocity+2*p2+2*p3+p4)/6;
        velocity+=h*(a+2*a2+2*a3+a4)/6;
      }
      return [position,velocity];
    }
    const [a,c]=advance(1,0), [b,d]=advance(0,1);
    const halfTrace=(a+d)/2, determinant=a*d-b*c;
    const discriminant=halfTrace**2-determinant;
    const radius=discriminant<=0?Math.sqrt(determinant):Math.abs(halfTrace)+Math.sqrt(discriminant);
    return Math.log(radius)/period;
  }

  function avoidedCrossing(coupling,rate) {
    const steps=1200, dt=20/(rate*steps);
    const startAngle=Math.atan2(2*coupling,-10);
    let reA=-Math.sin(startAngle/2), imA=0, reB=Math.cos(startAngle/2), imB=0;
    const series=[[ -10,reA*reA ]], ground=[[ -10,reA*reA ]];
    for(let j=1;j<=steps;j++) {
      const detuning=-10+20*(j-0.5)/steps, diagonal=detuning/2;
      const norm=Math.hypot(diagonal,coupling), scale=Math.sin(norm*dt)/norm, cosine=Math.cos(norm*dt);
      const [nextAr,nextAi]=[cosine*reA+scale*(diagonal*imA+coupling*imB),cosine*imA-scale*(diagonal*reA+coupling*reB)];
      const [nextBr,nextBi]=[cosine*reB+scale*(coupling*imA-diagonal*imB),cosine*imB-scale*(coupling*reA-diagonal*reB)];
      reA=nextAr; imA=nextAi; reB=nextBr; imB=nextBi;
      if(j%5===0) {
        const x=-10+20*j/steps, angle=Math.atan2(2*coupling,x);
        series.push([x,reA**2+imA**2]); ground.push([x,Math.sin(angle/2)**2]);
      }
    }
    return {series,ground,probability:reA**2+imA**2,norm:reA**2+imA**2+reB**2+imB**2};
  }

  function fixedDensityMu(temperature) {
    if(temperature===0) return 1;
    // The zero-temperature integral of 2 y² from zero to one is 2/3.
    const energyMax=1+35*temperature, end=Math.sqrt(energyMax), intervals=320, h=end/intervals;
    function count(mu) {
      let integral=0;
      for(let i=0;i<=intervals;i++) {
        const y=i*h, occupation=1/(1+Math.exp((y*y-mu)/temperature));
        integral+=(i===0 || i===intervals?1:i%2?4:2)*2*y*y*occupation;
      }
      return integral*h/3;
    }
    let low=-1, high=1.1;
    for(let iter=0;iter<36;iter++) {
      const mid=(low+high)/2;
      if(count(mid)>2/3) high=mid;
      else low=mid;
    }
    return (low+high)/2;
  }

  function stackReflectance(pairs,nHigh,wavelength) {
    const nLow=1.5, delta=pi*600/(2*wavelength);
    const c=Math.cos(delta),s=Math.sin(delta);
    let real=nLow, imag=0;
    for(let j=0;j<pairs;j++) {
      for(const index of [nLow,nHigh]) {
        // Y_in = (i n sinδ + cosδ Y_out)/(cosδ + i sinδ Y_out/n).
        const nr=c*real, ni=index*s+c*imag, dr=c-s*imag/index, di=s*real/index;
        const denominator=dr*dr+di*di;
        real=(nr*dr+ni*di)/denominator; imag=(ni*dr-nr*di)/denominator;
      }
    }
    return ((1-real)**2+imag**2)/((1+real)**2+imag**2);
  }

  Object.assign(window.fundamentalStudies ||= {}, {
    'ph-liouville': {
      controls:[['Time after release (s)',0,6,0.1,4],['Drag rate γ (1/s)',0,1,0.05,0.2]],
      xLabel:'Position q (m)',yLabel:'Momentum p (kg m/s)',
      calculate([t,gamma]) {
        const scale=Math.exp(-gamma*t), shear=gamma===0?t:-Math.expm1(-gamma*t)/gamma;
        const orbit=(name,s,pScale)=>({name,points:Array.from({length:181},(_,j)=>{
          const angle=2*pi*j/180,p=0.5*Math.sin(angle);
          return [Math.cos(angle)+s*p,pScale*p];
        })});
        return {curves:[orbit('Initial cloud',0,1),orbit('Free flight',t,1),orbit('With drag',shear,scale)],
          result:'Free position spread '+fmt(Math.hypot(1,t*0.5))+' m. The damped ellipse retains '+fmt(100*scale)+'% of its initial area and has position spread '+fmt(Math.hypot(1,0.5*shear))+' m.'};
      }
    },
    'ph-floquet': {
      controls:[['Fractional stiffness modulation',0,0.35,0.01,0.08],['Damping γ (1/s)',0,0.4,0.01,0.1],['Drive frequency Ω (rad/s)',18,22,0.05,20]],
      xLabel:'Drive angular frequency (rad/s)',yLabel:'Maximum growth exponent (1/s)',
      calculate([modulation,damping,drive]) {
        const averaged=x=>-damping+Math.sqrt(Math.max(0,(2.5*modulation)**2-(x/2-10)**2));
        const exact=x=>floquetExponent(modulation,damping,x), rate=exact(drive);
        return {curves:[sampled('One-period numerical map',18,22,112,exact),sampled('Weak-modulation average',18,22,112,averaged),mark('Selected exponent',drive,rate)],
          result:'At Ω = '+fmt(drive)+' rad/s, numerical exponent '+fmt(rate)+' s⁻¹ and averaged exponent '+fmt(averaged(drive))+' s⁻¹. '+(rate>1e-6?'Amplitude doubling time '+fmt(Math.log(2)/rate)+' s.':rate< -1e-6?'The largest mode decays.':'The mode is neutral to numerical precision.')};
      }
    },
    'ph-quench': {
      controls:[['Expansion ratio b',1,4,0.05,2],['Retained eigenmodes',3,70,1,12]],
      xLabel:'New mode index n',yLabel:'Probability',
      calculate([b,nmax]) {
        const each=[],cumulative=[];
        let sum=0, energy=0;
        for(let n=1;n<=nmax;n++) {
          const ratio=n/b;
          const amplitude=Math.abs(ratio-1)<1e-10?1/Math.sqrt(b):2*Math.sin(pi*ratio)/(pi*Math.sqrt(b)*(1-ratio**2));
          const p=amplitude**2;
          sum+=p; energy+=p*ratio**2; each.push([n,p]); cumulative.push([n,sum]);
        }
        return {curves:[{name:'Energy-measurement probability',points:each},{name:'Cumulative probability',points:cumulative}],
          result:'Probability retained '+fmt(sum)+', omitted '+fmt(Math.max(0,1-sum))+'. Energy sum '+fmt(energy)+' in units of the old ground energy. Exact infinite sum = 1.'};
      }
    },
    'ph-avoided-sweep': {
      controls:[['Coupling g (ℏ = 1)',0.1,1.2,0.05,0.5],['Sweep rate v (ℏ = 1)',0.25,4,0.05,1]],
      xLabel:'Detuning vt',yLabel:'Probability in first fixed basis state',
      calculate([g,v]) {
        const sweep=avoidedCrossing(g,v);
        return {curves:[{name:'Finite Schrödinger evolution',points:sweep.series},{name:'Instantaneous lower state',points:sweep.ground}],
          result:'Final fixed-basis probability '+fmt(sweep.probability)+', asymptotic Landau–Zener value '+fmt(Math.exp(-2*pi*g*g/v))+'. Propagated state norm squared '+fmt(sweep.norm)+'.'};
      }
    },
    'ph-sommerfeld': {
      controls:[['Reduced temperature T/TF',0.04,0.5,0.01,0.1]],xLabel:'Reduced temperature T/TF',yLabel:'Chemical potential μ/EF',
      calculate([temperature]) {
        const approx=t=>1-pi*pi*t*t/12, mu=fixedDensityMu(temperature);
        return {curves:[sampled('Fixed-density quadrature',0.04,0.5,46,fixedDensityMu),sampled('Quadratic approximation',0.04,0.5,46,approx),mark('Selected solution',temperature,mu)],
          result:'Fixed-number μ/EF = '+fmt(mu)+', quadratic prediction '+fmt(approx(temperature))+'. Leading heat capacity Cᵥ/(Nkᵦ) = '+fmt(pi*pi*temperature/2)+'.'};
      }
    },
    'ph-mean-field': {
      controls:[['Temperature T/Tc',0.45,1.5,0.01,0.8],['Field h/(kᵦTc)',-0.12,0.12,0.005,0]],
      xLabel:'Mean spin m',yLabel:'[f(m) − f(0)] / (kᵦTc)',
      calculate([t,field]) {
        const xlogx=x=>x===0?0:x*Math.log(x);
        const full=m=>-m*m/2-field*m+t*(xlogx((1+m)/2)+xlogx((1-m)/2)+Math.log(2));
        const quartic=m=>(t-1)*m*m/2+t*m**4/12-field*m;
        // For h ≠ 0 the global minimum has its sign; solve on this branch.
        let lo=0, hi=1;
        for(let j=0;j<52;j++) {
          const m=(lo+hi)/2;
          if(Math.tanh((m+Math.abs(field))/t)>m) lo=m;
          else hi=m;
        }
        const root=(field<0?-1:1)*(lo+hi)/2, m=Math.abs(root)<1e-5?0:root;
        const curves=[sampled('Full independent-neighbour free energy',-1,1,200,full),sampled('Quartic approximation',-1,1,200,quartic),mark('Equilibrium minimum',m,full(m))];
        if(field===0&&m>0) curves.push(mark('Symmetry-related minimum',-m,full(-m)));
        return {curves,result:'Equilibrium mean spin '+(field===0&&m>0?'±':'')+fmt(m)+'. Curvature at the minimum '+fmt(t/(1-m*m)-1)+' in units of kᵦTc.'};
      }
    },
    'ph-phase-filter': {
      controls:[['Specimen phase amplitude ε (rad)',0,0.8,0.02,0.1],['Central order shift φ (degrees)',-180,180,5,90]],
      xLabel:'Position in grating periods',yLabel:'Image intensity / mean incident intensity',
      calculate([epsilon,phase]) {
        const rad=phase*pi/180, steps=128;
        let central=0;
        for(let j=0;j<steps;j++) central+=Math.cos(epsilon*Math.cos(2*pi*j/steps))/steps;
        const exact=x=>{
          const theta=epsilon*Math.cos(2*pi*x);
          return (Math.cos(theta)+(Math.cos(rad)-1)*central)**2+(Math.sin(theta)+Math.sin(rad)*central)**2;
        };
        const linear=x=>1+2*epsilon*Math.sin(rad)*Math.cos(2*pi*x);
        const data=sampled('All grating orders',0,2,300,exact), values=data.points.map(pt=>pt[1]);
        return {curves:[data,sampled('First-order phase contrast',0,2,300,linear)],
          result:'Zero-order amplitude '+fmt(central)+'. Peak-to-trough intensity difference '+fmt(Math.max(...values)-Math.min(...values))+'; linear prediction '+fmt(4*epsilon*Math.abs(Math.sin(rad)))+'.'};
      }
    },
    'ph-bragg-stack': {
      controls:[['Quarter-wave pairs',1,12,1,4],['High layer index',1.5,2.5,0.05,2],['Probe wavelength (nm)',400,850,5,600]],
      xLabel:'Vacuum wavelength (nm)',yLabel:'Intensity reflectance',
      calculate([pairs,high,probe]) {
        const reflect=x=>stackReflectance(pairs,high,x), admittance=1.5*(high/1.5)**(2*pairs);
        return {curves:[sampled('Finite dielectric mirror',400,850,300,reflect),sampled('Bare air–substrate interface',400,850,1,()=>0.04),mark('Selected wavelength',probe,reflect(probe))],
          result:'At '+fmt(probe)+' nm reflectance is '+fmt(reflect(probe))+'. At 600 nm the pair-matrix formula gives '+fmt(((admittance-1)/(admittance+1))**2)+'.'};
      }
    },
    'ph-heat-transport': {
      controls:[['Filter centre (E − μ)/(kᵦT)',-4,4,0.1,1],['Filter width / (kᵦT)',0.2,8,0.1,2]],
      xLabel:'Excess energy (E − μ)/(kᵦT)',yLabel:'Normalised transport weight',
      calculate([centre,width]) {
        const weight=x=>1/(4*Math.cosh(x/2)**2), full=x=>weight(x)*Math.exp(-0.5*((x-centre)/width)**2);
        const intervals=600, h=40/intervals;
        let m0=0,m1=0,m2=0;
        for(let j=0;j<=intervals;j++) {
          const x=-20+j*h, quadrature=(j===0||j===intervals?1:j%2?4:2)*h/3;
          const w=full(x)*quadrature;
          m0+=w; m1+=w*x; m2+=w*x*x;
        }
        const mean=m1/m0,variance=m2/m0-mean**2;
        return {curves:[sampled('Energy-selected current carriers',-10,10,300,x=>full(x)/m0),sampled('Smooth metallic transport',-10,10,300,weight)],
          result:'Mean excess energy '+fmt(mean)+' kᵦT; electron Seebeck coefficient '+fmt(-86.1733*mean)+' µV/K. Open-circuit Lorenz ratio L/L₀ = '+fmt(3*variance/(pi*pi))+'.'};
      }
    },
    'ph-optical-phonon': {
      controls:[['Static relative permittivity',4.5,16,0.5,9],['Damping ratio γ/ωTO',0.01,0.30,0.01,0.05],['Probe frequency (THz)',1,14,0.1,7.5]],
      xLabel:'Optical frequency (THz)',yLabel:'Normal-incidence intensity reflectance',
      calculate([staticE,damping,probe]) {
        const background=4,to=6,lo=to*Math.sqrt(staticE/background);
        const reflect=f=>{
          const ratio=f/to, a=1-ratio**2,b=damping*ratio, denominator=a*a+b*b;
          const real=background+(staticE-background)*a/denominator,imag=(staticE-background)*b/denominator;
          const mod=Math.hypot(real,imag);
          const n=Math.sqrt(Math.max(0,(mod+real)/2)), k=Math.sqrt(Math.max(0,(mod-real)/2));
          return ((n-1)**2+k*k)/((n+1)**2+k*k);
        };
        return {curves:[sampled('Damped ionic crystal',0,14,350,reflect),sampled('Electronic background only',0,14,1,()=>1/9),mark('Selected frequency',probe,reflect(probe))],
          result:'Lossless TO pole '+fmt(to)+' THz and LO zero '+fmt(lo)+' THz. At '+fmt(probe)+' THz the chosen damping gives reflectance '+fmt(reflect(probe))+'.'};
      }
    }
  });
})();
