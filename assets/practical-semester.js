// Numerical experiments for the integrative problem sheets.
(() => {
  const studies = window.fundamentalStudies ||= {};
  function curve(name, min, max, fn, count = 120) {
    return {name, points: Array.from({length:count + 1}, (_, i) => {
      const x = min + i * (max - min) / count;
      return [x, fn(x)];
    })};
  }
  const fmt = x => Number(x.toPrecision(5)).toString();

  studies["semester-quantum-mechanics"] = {
    controls: [["Dipole coupling at the reference separation", .05, .9, .01, .4], ["Separation to inspect (R₀)", 1, 2.5, .02, 1.4]],
    xLabel: "Separation R/R₀", yLabel: "Energy change in units of ℏω₀",
    calculate([strength, radius]) {
      const coupling = r => strength / r**3;
      // Rationalising the sum avoids subtracting two nearly equal ground-state energies.
      const exact = r => {
        const l = coupling(r), root = Math.sqrt(1 - l*l), magnitude = Math.sqrt((1 + root)/2);
        return -l*l / (2 * (1 + root) * (1 + magnitude));
      };
      const quadratic = r => -(coupling(r)**2) / 8;
      return {curves: [curve("Exact pair",1,2.5,exact), curve("Second-order energy",1,2.5,quadratic),
        {name:"Selected measurement", points:[[radius, exact(radius)]]}],
        result:"At the selected separation the energy shift is " + fmt(exact(radius)) + " ℏω₀. Second order gives " + fmt(quadratic(radius)) + " ℏω₀."};
    }
  };
  studies["semester-classical-mechanics"] = {
    controls: [["Nominal middle-axis speed (rad/s)", 4, 16, .2, 12], ["Initial transverse speed (rad/s)", .02, .5, .01, .1]],
    xLabel: "Time (s)", yLabel: "First angular velocity component (rad/s)",
    calculate([spin, seed]) {
      const sigma = spin / Math.sqrt(15);
      const rhs = ([a,b,c]) => [-b*c/3, a*c/2, -a*b/5];
      const advance = (y, k, h) => y.map((value, i) => value + h*k[i]);
      let state = [seed, spin, 0];
      const points = [[0, seed]], h = .002;
      for (let j=1; j<=600; j++) {
        const k1 = rhs(state), k2 = rhs(advance(state,k1,h/2));
        const k3 = rhs(advance(state,k2,h/2)), k4 = rhs(advance(state,k3,h));
        state = state.map((v,i) => v + h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6);
        if (j%5===0) points.push([j*h,state[0]]);
      }
      return {curves:[{name:"Full Euler equations",points},curve("Linear perturbation",0,1.2,t=>seed*Math.cosh(sigma*t))],
        result:"Initial linear growth rate " + fmt(sigma) + " s⁻¹. With zero initial transverse slope the predicted first doubling time is " + fmt(Math.acosh(2)/sigma) + " s."};
    }
  };
  studies["semester-thermodynamics-statistical-physics"] = {
    controls:[["Atom number",1000,100000,1000,20000],["Isotropic trap frequency (Hz)",40,180,2,100]],
    xLabel:"Temperature T/Tc (continuum)", yLabel:"Excited cloud capacity / atom number",
    calculate([atoms,hz]) {
      const tc = Math.cbrt(atoms / 1.20205690316);
      function discrete(ratio) {
        const t = tc * ratio;
        // Beyond 42 thermal energies the remaining exponential tail has negligible weight.
        let sum = 0;
        for (let n=1;n<=Math.ceil(42*t);n++) sum += ((n+1)*(n+2)/2)/Math.expm1(n/t);
        return sum/atoms;
      }
      let lo=.4,hi=1.05;
      for(let i=0;i<38;i++){
        const mid=(lo+hi)/2;
        if(discrete(mid)>1) hi=mid; else lo=mid;
      }
      const tcNano = tc * 6.62607015e-34*hz/1.380649e-23*1e9;
      return {curves:[curve("Counted harmonic levels",.25,1.2,discrete,70),curve("Continuum density",.25,1.2,r=>r**3,70),
        curve("All atoms",.25,1.2,()=>1,2)], result:"Continuum Tc " + fmt(tcNano) + " nK. The discrete excited levels reach N at " + fmt(tcNano*(lo+hi)/2) + " nK."};
    }
  };
  studies["semester-condensed-matter"] = {
    controls:[["Longitudinal sound speed (m/s)", 4000,8500,50,6000],["Transverse sound speed (m/s)",1500,4000,50,3000]],
    xLabel:"Temperature (K)", yLabel:"Lattice heat capacity (J m⁻³ K⁻¹)",
    calculate([vl,vt]){
      const prefactor=2*Math.PI**2*(1.380649e-23)**4/(15*(1.054571817e-34)**3);
      const longitudinal=prefactor/vl**3, transverse=2*prefactor/vt**3;
      return {curves:[curve("All acoustic branches",0,6,t=>(longitudinal+transverse)*t**3),
        curve("Two transverse branches",0,6,t=>transverse*t**3),curve("Longitudinal branch",0,6,t=>longitudinal*t**3)],
        result:"β = " + fmt(longitudinal+transverse) + " J m⁻³ K⁻⁴. Transverse branches supply " + fmt(100*transverse/(longitudinal+transverse)) + "%."};
    }
  };
  studies["semester-waves-optics"] = {
    controls:[["Angular separation (milliarcseconds)",1,8,.05,3.1],["Intensity ratio",.1,1,.01,1]],
    xLabel:"Projected baseline (m)", yLabel:"Fringe visibility",
    calculate([angle,ratio]){
      const delta = angle*Math.PI/(180*3600000), wavelength=600e-9;
      const visibility = (b,r) => Math.hypot(1+r*Math.cos(2*Math.PI*b*delta/wavelength),r*Math.sin(2*Math.PI*b*delta/wavelength))/(1+r);
      return {curves:[curve("Selected binary",0,90,b=>visibility(b,ratio),350),
        curve("Equal intensities",0,90,b=>visibility(b,1),350)],
        result:"First minimum at " + fmt(wavelength/(2*delta)) + " m. Its intrinsic visibility is " + fmt((1-ratio)/(1+ratio)) + "."};
    }
  };
})();
