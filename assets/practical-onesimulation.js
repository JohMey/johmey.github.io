// The two teaching solvers below expose enough operations to check their OneSimulation counterparts line by line.
(() => {
  const studies=window.fundamentalStudies ||= {};
  studies['case-curvature']={
    controls:[['Section height (mm)',6,48,1,12],['Axial cells',6,48,2,24]],
    xLabel:'Position from clamp (mm)',yLabel:'Downward deflection (mm)',
    localModel:{file:'cantilever.py',args:([h,n])=>'--height '+h+' --cells '+n},
    calculate([height,n]){
      const L=.12,E=70e9,F=30,I=.018*(height/1000)**3/12,dx=L/n;
      let slope=0,w=0;
      const discrete=[[0,0]],exact=[[0,0]];
      for(let j=1;j<=n;j++){
        const x=j*dx;
        const curvatureLeft=F*(L-x+dx)/(E*I),curvatureRight=F*(L-x)/(E*I);
        const nextSlope=slope+dx*(curvatureLeft+curvatureRight)/2;
        w+=dx*(slope+nextSlope)/2;slope=nextSlope;
        discrete.push([1000*x,1000*w]);exact.push([1000*x,1000*F*x*x*(3*L-x)/(6*E*I)]);
      }
      return {curves:[{name:'Integrated curvature on the grid',points:discrete},{name:'Euler–Bernoulli solution',points:exact}],
        result:'Tip deflection '+(1000*w).toPrecision(5)+' mm on this 1D grid, '+exact[n][1].toPrecision(5)+' mm analytically. The OneSimulation model solves the same load on a 3D solid.'};
    }
  };
  studies['case-oscillator']={
    controls:[['Damping / critical damping',0,4,.02,.2]],
    xLabel:'Time after release (s)',yLabel:'Displacement (mm)',
    localModel:{file:'oscillator.py',args:([z])=>'--ratio '+z},
    calculate([z]){
      const dt=.01,f=(x,v)=>-4*x-4*z*v,points=[[0,10]],exact=[[0,10]];
      let x=10,v=0,maxError=0;
      for(let j=1;j<=600;j++){
        const a=f(x,v), v2=v+a*dt/2,a2=f(x+v*dt/2,v2),
          v3=v+a2*dt/2,a3=f(x+v2*dt/2,v3),v4=v+a3*dt,a4=f(x+v3*dt,v4);
        x+=dt*(v+2*v2+2*v3+v4)/6;v+=dt*(a+2*a2+2*a3+a4)/6;
        const t=j*dt,u=2*t;
        let y;
        if(z<1){const b=Math.sqrt(1-z*z);y=Math.exp(-z*u)*(Math.cos(b*u)+z*Math.sin(b*u)/b);}
        else if(z===1)y=(1+u)*Math.exp(-u);
        else {const b=Math.sqrt(z*z-1),s=-1/(z+b),q=-z-b;y=(-q*Math.exp(s*u)+s*Math.exp(q*u))/(s-q);}
        y*=10;maxError=Math.max(maxError,Math.abs(y-x));points.push([t,x]);exact.push([t,y]);
      }
      return {curves:[{name:'Fourth-order Runge–Kutta',points},{name:'Characteristic-root solution',points:exact}],
        result:'At 6 s the displacement is '+x.toPrecision(5)+' mm. Largest difference from the exact response on 600 steps: '+maxError.toExponential(2)+' mm.'};
    }
  };
})();
