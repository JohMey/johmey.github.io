(() => {
  'use strict';
  const studies = window.fundamentalStudies ||= {};
  const pi = Math.PI;
  function trace(name,a,b,fn,n=160) {
    return {name,points:Array.from({length:n+1},(_,i)=>{
      const x=a+(b-a)*i/n;
      return [x,fn(x)];
    })};
  }
  studies['semester-engineering-curved-beam'] = {
    controls:[['Inner radius (mm)',15,100,1,30],['Section depth (mm)',5,35,1,20],['Applied moment (N m)',4,20,1,12]],
    xLabel:'Distance from inner face (mm)',yLabel:'Circumferential stress (MPa)',
    calculate([inner,depth,moment]) {
      const r=inner*1e-3,d=depth*1e-3,b=0.010,cent=r+d/2;
      const neutral=d/Math.log1p(d/r), e=cent-neutral;
      const stress=x=>moment/(b*d*e)*(neutral/(r+x*1e-3)-1)/1e6;
      const straight=x=>moment*(d/2-x*1e-3)/(b*d**3/12)/1e6;
      return {curves:[trace('Curved beam',0,depth,stress),trace('Straight-beam approximation',0,depth,straight)],
        result:'The neutral radius is '+(1000*neutral).toFixed(2)+' mm and peak inner stress is '+stress(0).toFixed(2)+' MPa, compared with '+straight(0).toFixed(2)+' MPa for a straight beam.'};
    }
  };
  studies['semester-engineering-couette'] = {
    controls:[['Outer radius (mm)',16,30,0.5,20],['Inner rotation (rad/s)',1,12,0.5,5],['Viscosity (Pa s)',0.05,0.6,0.01,0.25]],
    xLabel:'Radius (mm)',yLabel:'Azimuthal fluid velocity (m/s)',
    calculate([outer,omega,viscosity]) {
      const a=0.015,b=outer/1000,L=0.08;
      const torque=4*pi*viscosity*L*omega*a*a*b*b/(b*b-a*a);
      return {curves:[trace('Cylindrical solution',15,outer,x=>omega*a*a/(b*b-a*a)*(b*b/(x/1000)-x/1000)),
        trace('Planar linear profile',15,outer,x=>omega*a*(outer-x)/(outer-15))],
        result:'The lateral torque is '+(torque*1000).toFixed(3)+' mN m and inner wall shear magnitude is '+(torque/(2*pi*L*a*a)).toFixed(2)+' Pa. End effects are excluded.'};
    }
  };
  studies['semester-engineering-calorimeter'] = {
    controls:[['Hot body capacity (J/K)',60,200,5,120],['Cold body capacity (J/K)',150,450,10,300],['Contact conductance (W/K)',1,6,0.1,2.5]],
    xLabel:'Time after contact (s)',yLabel:'Body temperature (°C)',
    calculate([hot,cold,H]) {
      const mean=(70*hot+20*cold)/(hot+cold),tau=hot*cold/((hot+cold)*H);
      const one=t=>mean+50*cold/(hot+cold)*Math.exp(-t/tau);
      const two=t=>mean-50*hot/(hot+cold)*Math.exp(-t/tau);
      return {curves:[trace('Initially hot',0,160,one),trace('Initially cold',0,160,two),trace('Energy-weighted mean',0,160,()=>mean)],
        result:'The isolated common temperature is '+mean.toFixed(2)+' °C. The difference decays with time constant '+tau.toFixed(2)+' s; after 40 s the readings are '+one(40).toFixed(2)+' and '+two(40).toFixed(2)+' °C.'};
    }
  };
  studies['semester-engineering-vibration-absorber'] = {
    controls:[['Absorber / primary mass',0.04,0.24,0.01,0.10],['Absorber tuning ratio',0.80,1.20,0.01,1.00],
      ['Absorber damping ratio',0,0.2,0.01,0.04]],
    xLabel:'Forcing frequency / primary natural frequency',yLabel:'Amplitude / primary static deflection',
    calculate([mu,tuning,za]) {
      const primaryDamping=0.03;
      function amplitudes(r) {
        const pr=1-r*r,pi2=2*primaryDamping*r,ar=tuning*tuning,ai=2*za*tuning*r;
        const dr=pr*(ar-r*r)-pi2*ai-mu*r*r*ar,di=pr*ai+pi2*(ar-r*r)-mu*r*r*ai;
        const den=Math.hypot(dr,di);
        return [Math.hypot(ar-r*r,ai)/den,r*r/den];
      }
      return {curves:[trace('Primary motion',0.55,1.5,r=>amplitudes(r)[0],400),
        trace('Absorber relative travel',0.55,1.5,r=>amplitudes(r)[1],400)],
        result:'At the bare primary frequency the primary amplitude is '+amplitudes(1)[0].toFixed(2)+' times static deflection. Relative absorber travel is '+amplitudes(1)[1].toFixed(2)+' times static deflection.'};
    }
  };
  studies['semester-engineering-planetary'] = {
    controls:[['Sun teeth',24,40,2,30],['Planet teeth',15,35,1,20],['Sun speed (rpm)',300,1800,50,1200]],
    xLabel:'Ring speed (rpm)',yLabel:'Member speed (rpm)',
    calculate([sun,planet,sunSpeed]) {
      const ring=sun+2*planet;
      const carrier=r=>(sun*sunSpeed+ring*r)/(sun+ring);
      const spin=r=>carrier(r)-sun/planet*(sunSpeed-carrier(r));
      return {curves:[trace('Carrier',-600,600,carrier),trace('Planet spin in stationary frame',-600,600,spin)],
        result:'The ring has '+ring+' teeth. With a fixed ring the carrier runs at '+carrier(0).toFixed(1)+' rpm and the planet spins at '+spin(0).toFixed(1)+' rpm in the stationary frame.'};
    }
  };
  studies['semester-engineering-antiwindup'] = {
    controls:[['High reference duration (s)',2,12,0.5,6],['Tracking time (s)',0.1,3,0.1,0.5],['Actuator maximum',0.6,1.4,0.1,1]],
    xLabel:'Time (s)',yLabel:'Normalised plant output or reference',
    calculate([duration,tracking,limit]) {
      const end=duration+10,step=0.002, points=[[],[],[]],state=[[0,0],[0,0]];
      let plain=0,controlled=0;
      for(let i=0;i<=Math.round(end/step);i++) {
        const t=i*step,ref=t<duration?2:0.4;
        if(i%20===0){for(let j=0;j<2;j++)points[j].push([t,state[j][0]]);points[2].push([t,ref]);}
        for(let j=0;j<2;j++){
          const [y,acc]=state[j],err=ref-y,requested=2*err+acc;
          const applied=Math.min(limit,Math.max(0,requested));
          if(t>=duration&&y>0.6 && applied>=limit-1e-9){if(j===0)plain+=step;else controlled+=step;}
          state[j]=[y+(applied-y)*(1-Math.exp(-step)),acc+step*(err+(j===1?(applied-requested)/tracking:0))];
        }
      }
      return {curves:[{name:'Ordinary PI',points:points[0]},{name:'PI with tracking',points:points[1]},{name:'Reference',points:points[2]}],
        result:'After the reference drops, time still at the upper limit while output exceeds 0.6 is '+plain.toFixed(2)+' s for ordinary PI and '+controlled.toFixed(2)+' s with tracking. The plant time constant is 1 s.'};
    }
  };
})();
