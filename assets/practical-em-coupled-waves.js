(() => {
  'use strict';
  const studies = window.fundamentalStudies ||= {};
  const pi = Math.PI;
  const c = 299792458;
  const h = 6.62607015e-34;
  const kB = 1.380649e-23;

  function curve(name, start, stop, fn, intervals = 240) {
    return {name, points: Array.from({length: intervals + 1}, (_, i) => {
      const x = start + (stop-start)*i/intervals;
      return [x, fn(x)];
    })};
  }

  studies['em-coupled-waves-trap'] = {
    controls: [['Bead radius (nm)',25,55,1,40],['Optical power (mW)',30,180,5,90],['Beam waist (µm)',0.55,0.90,0.01,0.65]],
    xLabel: 'Lateral displacement (µm)', yLabel: 'Potential above trap minimum (thermal energies)',
    calculate([radius,power,waist]) {
      const a = radius*1e-9, w = waist*1e-6;
      const liquidIndex = 1.33, beadIndex = 1.59, thermalEnergy = kB*298;
      const contrast = (beadIndex**2-liquidIndex**2)/(beadIndex**2+2*liquidIndex**2);
      const intensity = 2*power*1e-3/(pi*w*w);
      const depth = 2*pi*a**3*liquidIndex*contrast*intensity/c;
      const stiffness = 4*depth/w**2, drag = 6*pi*0.001*a, reducedDepth = depth/thermalEnergy;
      return {curves: [
        curve('Gaussian well',-waist,waist,x=>reducedDepth*(1-Math.exp(-2*(x/waist)**2))),
        curve('Local harmonic model',-waist,waist,x=>2*reducedDepth*(x/waist)**2),
        curve('One thermal energy',-waist,waist,()=>1)],
        result:'The well depth is '+reducedDepth.toFixed(2)+' kBT. The local model gives an rms displacement of '+(1e9*Math.sqrt(thermalEnergy/stiffness)).toFixed(1)+' nm and a corner frequency of '+(stiffness/(2*pi*drag)).toFixed(1)+' Hz at 298 K.'};
    }
  };
  studies['em-coupled-waves-mode-splitting'] = {
    controls: [['Coupling G/(2π) (kHz)',0,350,5,160],['Optical linewidth κ/(2π) (kHz)',60,400,10,240],['Mechanical detuning (kHz)',-300,300,10,0]],
    xLabel: 'Probe offset from uncoupled optical sideband (kHz)', yLabel: 'Mode energy relative to bare optical peak',
    calculate([coupling,width,detuning]) {
      const opticalHalf = width/2, mechanicalHalf = 20;
      function intensities(x) {
        const y = x-detuning;
        const denominator = (opticalHalf*mechanicalHalf-x*y+coupling**2)**2
          +(opticalHalf*y+mechanicalHalf*x)**2;
        return [opticalHalf**2*(mechanicalHalf**2+y*y)/denominator,
          opticalHalf**2*coupling**2/denominator];
      }
      const discriminant = coupling**2-((width-40)/4)**2;
      const splitting = 2*Math.sqrt(Math.max(0,discriminant));
      const cooperativity = 4*coupling**2/(width*40);
      return {curves:[curve('Optical mode',-600,600,x=>intensities(x)[0],600),
        curve('Mechanical mode',-600,600,x=>intensities(x)[1],600)],
        result:'At zero bare detuning the real pole separation is '+splitting.toFixed(1)+' kHz. For the selected coupling and linewidths the cooperativity is '+cooperativity.toFixed(2)+'. The mechanical linewidth is fixed at 40 kHz.'};
    }
  };
  studies['em-coupled-waves-blackbody'] = {
    controls: [['Source temperature (K)',300,700,5,400],['Ambient temperature (K)',270,330,5,295],['Collection half-angle (°)',2,30,1,10]],
    xLabel:'Vacuum wavelength (µm)',yLabel:'Collected power per wavelength (µW/µm)',
    calculate([temperature,ambient,angle]) {
      const throughput = 1e-6*pi*Math.sin(angle*pi/180)**2*0.65;
      function spectral(lambdaUm,T) {
        const wavelength = lambdaUm*1e-6;
        return throughput*(2*h*c*c/wavelength**5)/Math.expm1(h*c/(wavelength*kB*T));
      }
      const net = x=>spectral(x,temperature)-spectral(x,ambient);
      const bands = 240, step = 4/bands;
      let sum = (net(8)+net(12))/2;
      for (let i=1;i<bands;i++) sum+=net(8+i*step);
      return {curves:[curve('Source',5,20,x=>spectral(x,temperature)),curve('Ambient',5,20,x=>spectral(x,ambient)),
        curve('Source minus ambient',5,20,net)],
        result:'The net 8–12 µm signal is '+(sum*step).toFixed(3)+' µW for a 1 mm² area and 65% throughput. Ideal source peak wavelength is '+(2897.77/temperature).toFixed(2)+' µm.'};
    }
  };
  studies['em-coupled-waves-doppler'] = {
    controls:[['Mirror speed, positive receding (mm/s)',-5,5,0.1,2],['Vacuum wavelength (nm)',500,800,1,633],['Fringe visibility',0.1,1,0.05,0.8]],
    xLabel:'Detector time (ms)',yLabel:'Normalised quadrature signal',
    calculate([speed,wavelength,visibility]) {
      const velocity = speed*1e-3, beta = velocity/c;
      const beat = 2*velocity/(wavelength*1e-9*(1+beta));
      const phase = t=>2*pi*beat*t*1e-3;
      return {curves:[curve('Cosine channel',0,1,t=>visibility*Math.cos(phase(t)),600),
        curve('Sine channel',0,1,t=>visibility*Math.sin(phase(t)),600)],
        result:'The signed beat is '+beat.toFixed(2)+' Hz. In 1 ms the mirror travels '+speed.toFixed(2)+' µm and the unwrapped phase changes by '+(2*pi*beat*1e-3).toFixed(2)+' rad.'};
    }
  };
})();
