(() => {
  const fmt = n => Number(n.toPrecision(4)).toString();
  const logError = (error, floor) => Math.log10(Math.max(floor,Math.abs(error)));

  function rotate([q,p], real, imag) {
    return [real*q-imag*p, imag*q+real*p];
  }

  function finiteDifferenceLayer(lambda, n) {
    const h = 1/n, main = 2+(lambda*h)**2, right = Math.exp(-lambda);
    const diagonal = Array(n-1).fill(main), rhs = Array(n-1).fill(0);
    rhs[0] += 1;
    rhs[n-2] += right;
    for (let j=1;j<n-1;j++) {
      diagonal[j] -= 1/diagonal[j-1];
      rhs[j] += rhs[j-1]/diagonal[j-1];
    }
    const values = Array(n+1);
    values[0] = 1;
    values[n] = right;
    values[n-1] = rhs[n-2]/diagonal[n-2];
    for (let j=n-2;j>=1;j--) values[j] = (rhs[j-1]+values[j+1])/diagonal[j-1];
    return values;
  }

  Object.assign(window.fundamentalStudies ||= {}, {
    'num-ode-midpoint': {
      controls: [['Time step h (ω = 1)',0.05,0.8,0.05,0.5],['Number of forward steps',5,60,1,20]],
      xLabel:'Scaled time ωt', yLabel:'Oscillator position q',
      calculate([h,n]) {
        const half = h/2, realMid = (1-half*half)/(1+half*half), imagMid = -h/(1+half*half);
        const realRK = 1-h*h/2, imagRK = -h;
        const pointsMid=[[0,1]], pointsRK=[[0,1]];
        let midpoint=[1,0], rk=[1,0];
        for (let j=1;j<=n;j++) {
          midpoint=rotate(midpoint,realMid,imagMid);
          rk=rotate(rk,realRK,imagRK);
          pointsMid.push([j*h,midpoint[0]]);
          pointsRK.push([j*h,rk[0]]);
        }
        for (let j=0;j<n;j++) {
          midpoint=rotate(midpoint,realMid,-imagMid);
          rk=rotate(rk,realRK,-imagRK);
        }
        const distance = ([q,p])=>Math.hypot(q-1,p);
        const exact = Array.from({length:8*n+1},(_,j)=>[h*j/8, Math.cos(h*j/8)]);
        return {curves:[{name:'Exact',points:exact},{name:'Implicit midpoint',points:pointsMid},{name:'Explicit midpoint RK2',points:pointsRK}],
          result:'Implicit midpoint phase lag '+fmt(n*(h-2*Math.atan(half)))+' rad. Forward-backward distance from the start is '+fmt(distance(midpoint))+' for implicit midpoint and '+fmt(distance(rk))+' for explicit midpoint.'};
      }
    },
    'num-ode-imex': {
      controls: [['Relaxation time ε',0.005,0.10,0.005,0.02],['Steps to final time 1',10,80,1,20]],
      xLabel:'Time t',yLabel:'log₁₀ absolute solution error',
      calculate([epsilon,n]) {
        const h=1/n, ratio=h/epsilon, implicitPoints=[[0,-15]], explicitPoints=[[0,-15]];
        let imex=1, explicit=1, worst=0, last=0;
        for(let j=1;j<=n;j++) {
          const t=(j-1)*h, next=j*h;
          imex=(imex+h*Math.cos(t)+ratio*Math.sin(next))/(1+ratio);
          explicit+=h*Math.cos(t)+ratio*(Math.sin(t)-explicit);
          const exact=Math.sin(next)+Math.exp(-next/epsilon);
          last=Math.abs(imex-exact);
          worst=Math.max(worst,last);
          implicitPoints.push([next,logError(last,1e-15)]);
          explicitPoints.push([next,logError(explicit-exact,1e-15)]);
        }
        return {curves:[{name:'Implicit-explicit Euler',points:implicitPoints},{name:'Explicit Euler',points:explicitPoints}],
          result:'h/ε = '+fmt(ratio)+'. Explicit perturbation factor '+fmt(1-ratio)+', implicit factor '+fmt(1/(1+ratio))+'. IMEX maximum error '+fmt(worst)+', final error '+fmt(last)+'.'};
      }
    },
    'num-ode-bvp': {
      controls: [['Layer parameter λ',2,40,1,20],['Number of mesh intervals',2,128,2,32]],
      xLabel:'Position x',yLabel:'log₁₀ absolute solution error',
      calculate([lambda,n]) {
        const layer=finiteDifferenceLayer(lambda,n), nodeErrors=[], shootingErrors=[];
        let maximum=0;
        layer.forEach((value,j)=>{
          const x=j/n, error=Math.abs(value-Math.exp(-lambda*x));
          maximum=Math.max(maximum,error);
          nodeErrors.push([x,logError(error,1e-16)]);
        });
        for(let j=0;j<=240;j++) {
          const x=j/240;
          shootingErrors.push([x,logError(1e-12*Math.sinh(lambda*x)/lambda,1e-16)]);
        }
        return {curves:[{name:'Finite-difference collocation',points:nodeErrors},{name:'Propagated slope error',points:shootingErrors}],
          result:'λh = '+fmt(lambda/n)+'. Maximum collocation error '+fmt(maximum)+'. The slope perturbation produces '+fmt(1e-12*Math.sinh(lambda)/lambda)+' right-face error.'};
      }
    }
  });
})();
