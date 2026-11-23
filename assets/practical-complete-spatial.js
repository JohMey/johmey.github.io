// Boundary constructions: each formula is stated and derived in the accompanying chapter.
(() => {
  const registry=window.fundamentalStudies ||= {};
  const grid=(a,b,n=160)=>Array.from({length:n+1},(_,i)=>a+(b-a)*i/n);
  const curve=(name,x,f)=>({name,points:x.map(t=>[t,f(t)])});
  const add=(key,controls,xLabel,yLabel,calculate)=>registry['complete-spatial-'+key]={controls,xLabel,yLabel,calculate};
  const fmt=x=>Number(x.toPrecision(5)).toString();
  const norm=p=>Math.hypot(...p),sub=(a,b)=>a.map((v,i)=>v-b[i]);
  function fibonacci(n,offset=0){return Array.from({length:n},(_,i)=>{const z=1-2*(i+.5)/n,p=i*Math.PI*(3-Math.sqrt(5))+offset;return [Math.sqrt(1-z*z)*Math.cos(p),Math.sqrt(1-z*z)*Math.sin(p),z];});}
  function solve(matrix,rhs){
    const a=matrix.map((r,i)=>[...r,rhs[i]]),n=a.length;
    for(let k=0;k<n;k++){
      let p=k;for(let j=k+1;j<n;j++)if(Math.abs(a[j][k])>Math.abs(a[p][k]))p=j;
      [a[k],a[p]]=[a[p],a[k]];
      for(let j=k+1;j<n;j++){const f=a[j][k]/a[k][k];for(let i=k+1;i<=n;i++)a[j][i]-=f*a[k][i];a[j][k]=0;}
    }
    const x=Array(n).fill(0);for(let j=n-1;j>=0;j--){let s=a[j][n];for(let i=j+1;i<n;i++)s-=a[j][i]*x[i];x[j]=s/a[j][j];}return x;
  }
  add('sphere', [['Number of source points',24,96,24,48],['Charge distance d/a',1.5,3,.05,2],['Source radius / a',.45,.85,.05,.65]],
    'Polar angle θ on sphere (degrees)','Boundary potential error (V)',([n,d,r])=>{
      const boundary=fibonacci(n),sources=boundary.map(p=>p.map(x=>x*r)),charge=[0,0,d];
      const matrix=boundary.map(p=>sources.map(s=>1/norm(sub(p,s)))),rhs=boundary.map(p=>-1/norm(sub(p,charge)));
      const weights=solve(matrix,rhs),factor=8.9875517923;
      const potential=p=>factor*(1/norm(sub(p,charge))+sources.reduce((v,s,i)=>v+weights[i]/norm(sub(p,s)),0));
      const field=p=>{
        const real=sub(p,charge),e=real.map(v=>v/norm(real)**3);
        sources.forEach((s,i)=>{const delta=sub(p,s),k=weights[i]/norm(delta)**3;for(let j=0;j<3;j++)e[j]+=k*delta[j];});return e;
      };
      const withheld=Math.max(...fibonacci(1201,.37).map(p=>Math.abs(potential(p))));
      const residual=Math.max(...matrix.map((row,j)=>Math.abs(row.reduce((v,a,i)=>v+a*weights[i],0)-rhs[j])));
      const lines=[],faces=[];
      const point=(theta,phi)=>[Math.sin(theta)*Math.cos(phi),Math.sin(theta)*Math.sin(phi),Math.cos(theta)];
      for(let i=0;i<18;i++)for(let j=0;j<30;j++)faces.push({vertices:[point(i*Math.PI/18,j*2*Math.PI/30),point((i+1)*Math.PI/18,j*2*Math.PI/30),point((i+1)*Math.PI/18,(j+1)*2*Math.PI/30),point(i*Math.PI/18,(j+1)*2*Math.PI/30)],color:'#94b8af'});
      for(const angle of [Math.PI/6,Math.PI/2,5*Math.PI/6,Math.PI])for(const phi of [0,Math.PI/2,Math.PI,3*Math.PI/2]){
        let p=point(angle,phi).map((v,i)=>charge[i]+.08*v),points=[p];
        for(let step=0;step<160;step++){const e=field(p),length=norm(e);p=p.map((v,i)=>v+.035*e[i]/length);points.push(p);if(norm(p)<1.015||norm(p)>4.4)break;}
        lines.push({points,color:'#ad491d',width:1.1});
      }
      const angles=grid(0,180),sum=weights.reduce((a,b)=>a+b,0);
      return {curves:[curve('Computed surface, φ=0°',angles,t=>potential(point(t*Math.PI/180,0))),curve('Computed surface, φ=90°',angles,t=>potential(point(t*Math.PI/180,Math.PI/2))),curve('Exact image construction',angles,()=>0)],
        result:`Linear residual ${residual.toExponential(2)}; maximum error on 1201 withheld surface points ${fmt(withheld)} V. Induced charge ${fmt(sum)} nC; exact ${fmt(-1/d)} nC.`,
        scene:{faces,lines,points:[{position:charge,label:'+1 nC',color:'#ad491d'},{position:[0,0,0],label:'Grounded sphere'}],axisLabels:['x/a','y/a','z/a'],caption:'Numerically traced electric-field lines outside a grounded sphere (a=1 m). The computed field uses fitted interior sources; slight loss of axial symmetry is approximation error. The fictitious interior charges are not physical charges in the metal.'}};
    });
  registry['complete-spatial-sphere'].localModel={file:'complete-spatial-sphere.py',url:'/downloads/fundamentals/examples/complete-spatial-sphere.py',args:([n,d,r])=>`--sources ${n} --distance ${d} --source-radius ${r}`};

  add('wedge', [['Wedge denominator m',2,8,1,3],['Radial source distance (m)',.5,2,.1,1],['Selected fraction of opening',.1,.9,.05,.5]],
    'Position inside opening φ₀/α','Force on 1 nC source (nN)',([m,r,selected])=>{
      function force(f){const alpha=Math.PI/m,phi=f*alpha,p=[r*Math.cos(phi),r*Math.sin(phi)],E=[0,0];
        for(let j=0;j<m;j++)for(const sign of [-1,1]){if(sign===1&&j===0)continue;const t=sign*phi+2*j*alpha,s=[r*Math.cos(t),r*Math.sin(t)],v=sub(p,s),den=norm(v)**3;E[0]+=8.9875517923*sign*v[0]/den;E[1]+=8.9875517923*sign*v[1]/den;}
        return [E[0]*Math.cos(phi)+E[1]*Math.sin(phi),-E[0]*Math.sin(phi)+E[1]*Math.cos(phi)];}
      const x=grid(.05,.95),f=force(selected);
      return {curves:[curve('Radial component',x,u=>force(u)[0]),curve('Azimuthal component',x,u=>force(u)[1]),{name:'Selected radial force',points:[[selected,f[0]]]},{name:'Selected azimuthal force',points:[[selected,f[1]]]}],result:`Opening ${fmt(180/m)}°; ${2*m-1} fictitious images. At φ₀/α=${selected}, radial force ${fmt(f[0])} nN and azimuthal force ${fmt(f[1])} nN. On the bisector the azimuthal component vanishes.`};
    });
  add('cylinder', [['Cylinder radius a (mm)',1,20,1,10],['Line charge (nC/m)',.1,2,.1,1],['Selected d/a',1.1,5,.1,2.5]],
    'Source distance d/a','Attraction per length (µN/m)',([a,q,selected])=>{
      const k=17.9751035846*q*q/a,x=grid(1.1,5),ground=s=>k*s/(s*s-1),isolated=s=>k/(s*(s*s-1));
      return {curves:[curve('Grounded conductor',x,ground),curve('Neutral isolated conductor',x,isolated),{name:'Selected grounded force',points:[[selected,ground(selected)]]},{name:'Selected isolated force',points:[[selected,isolated(selected)]]}],result:`At d=${fmt(a*selected)} mm: grounded attraction ${fmt(ground(selected))} µN/m, isolated attraction ${fmt(isolated(selected))} µN/m. The image lies ${fmt(a/selected)} mm from the axis; electrical preparation changes the distant force law.`};
    });
  add('interface', [['Permittivity ratio ε₂/ε₁',.2,8,.1,.5],['Charge q (pC)',1,20,1,10],['Selected height d (mm)',.2,5,.1,1]],
    'Height above interface (mm)','Normal force (nN; positive upwards)',([ratio,q,d])=>{
      const eta=(1-ratio)/(1+ratio),k=.56172198701875*q*q,x=grid(.2,5);
      return {curves:[curve('Dielectric interface',x,t=>k*eta/t**2),curve('Grounded-plane limit',x,t=>-k/t**2),{name:'Selected dielectric force',points:[[d,k*eta/d**2]]}],result:`With ε₁=4ε₀, q′/q=${fmt(eta)} and q″/q=${fmt(2*ratio/(1+ratio))}. Force at ${d} mm is ${fmt(k*eta/d**2)} nN: ${eta>0?'repulsive':eta<0?'attractive':'zero'}.`};
    });
  add('two-wires', [['Relative permittivity',1,6,.1,1],['Selected D/(2a)',1.1,12,.1,2]],
    'Centre spacing / diameter D/(2a)','Pair capacitance per length (pF/m)',([er,s])=>{
      const c=27.816251386*er,x=grid(1.05,12);
      return {curves:[curve('Exact bipolar-coordinate result',x,t=>c/Math.acosh(t)),curve('Widely separated approximation',x,t=>c/Math.log(2*t)),{name:'Selected spacing',points:[[s,c/Math.acosh(s)]]}],result:`At D/(2a)=${s}, C′=${fmt(c/Math.acosh(s))} pF/m. For a homogeneous nonmagnetic dielectric the corresponding TEM impedance is ${fmt(119.9169832*Math.acosh(s)/Math.sqrt(er))} Ω.`};
    });
  add('parallel-planes', [['Included image-pair order',1,40,1,12],['Selected charge position s/L',.1,.9,.05,.5]],
    'Charge position s/L','Force / [q²/(4πεL²)]',([order,s])=>{
      const force=u=>{let f=0;for(let n=-order;n<=order+1;n++){const delta=2*u-2*n;f-=delta/Math.abs(delta)**3;}return f;};
      const nearest=u=>.25*(1/(1-u)**2-1/u**2),x=grid(.05,.95);
      return {curves:[curve('Symmetrically truncated image sum',x,force),curve('Nearest image at each face',x,nearest),{name:'Selected position',points:[[s,force(s)]]}],result:`At s/L=${s}, normalised force=${fmt(force(s))}. Induced charge fractions: lower plate ${fmt(-(1-s))}, upper plate ${fmt(-s)}. At the midpoint force cancels, while both plates carry −q/2.`};
    });
  add('buried-plane', [['Permittivity ratio ε₂/ε₁',.2,4,.1,.5],['Film thickness (mm)',.1,3,.1,1]],
    'Pattern wavelength (mm)','Reflected / incident potential',([ratio,t])=>{
      const x=grid(.2,30),f=lambda=>{const u=ratio/Math.tanh(2*Math.PI*t/lambda);return (1-u)/(1+u);};
      return {curves:[curve('Dielectric film over grounded metal',x,f),curve('Semi-infinite dielectric',x,()=>(1-ratio)/(1+ratio))],result:ratio<1?`The reflection changes sign at wavelength ${fmt(2*Math.PI*t/Math.atanh(ratio))} mm. Long wavelengths see the buried ground; short wavelengths see the dielectric interface.`:'The reflection stays negative. When both permittivities agree, it equals −exp(−4πt/λ).'};
    });
})();
