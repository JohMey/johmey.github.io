(() => {
  const studies = window.fundamentalStudies ||= {};
  studies['getting-started-reaction'] = {
    controls: [['Rate k (1/s)', .1, 3, .1, .8], ['Initial A (mol/m³)', .2, 3, .1, 1], ['Time step (s)', .0125, 1.25, .0125, .0625]],
    xLabel: 'Time (s)', yLabel: 'Concentration (mol/m³)',
    localModel: {file:'get-started-reaction.py',url:'/downloads/get-started-reaction.py',args:([k,c,h])=>`--rate ${k} --initial ${c} --step ${h} --end 6.25`},
    calculate([k,c,h]) {
      let a=c,b=0,t=0,error=0,balance=0;
      const A=[[0,c]],B=[[0,0]],EA=[[0,c]],EB=[[0,0]];
      // Classical RK4, as in the downloadable OneSimulation model.
      while(t<6.25-1e-12) {
        const step=Math.min(h,6.25-t);
        const s1=-k*a,s2=-k*(a+step*s1/2),s3=-k*(a+step*s2/2),s4=-k*(a+step*s3);
        const change=step*(s1+2*s2+2*s3+s4)/6;
        a+=change;b-=change;t+=step;
        const exact=c*Math.exp(-k*t);
        A.push([t,a]);B.push([t,b]);
        error=Math.max(error,Math.abs(a-exact),Math.abs(b-(c-exact)));
        balance=Math.max(balance,Math.abs(a+b-c));
      }
      for(let j=1;j<=300;j++){const x=j*6.25/300,e=c*Math.exp(-k*x);EA.push([x,e]);EB.push([x,c-e]);}
      return {curves:[{name:'A · RK4',points:A},{name:'B · RK4',points:B},{name:'A · exact',points:EA},{name:'B · exact',points:EB}],
        result:`Half-life ${(Math.log(2)/k).toPrecision(4)} s; kh = ${(k*h).toPrecision(4)}. Maximum nodal error ${error.toExponential(3)} mol/m³; concentration-sum error ${balance.toExponential(2)} mol/m³.`+(k*h>2.785?' The step exceeds the RK4 negative-real stability interval. This raw RK4 preview exposes growth and negative B; OneSimulation rejects the negative concentration.':'')};
    }
  };
})();
