// Browser calculations for the particular examples beside each widget.
// Independent OneSimulation notebooks remain linked at the point of use.
(() => {
  const studies=window.fundamentalStudies ||= {};
  const xs=(a,b,n=120)=>Array.from({length:n+1},(_,i)=>a+(b-a)*i/n);
  const curve=(name,x,f)=>({name,points:x.map((t,i)=>[t,f(t,i)])});
  const fmt=x=>Number(x.toPrecision(5)).toString();
  const log=x=>Math.log10(Math.max(1e-16,Math.abs(x)));
  const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
  const norm=a=>Math.hypot(...a);
  const add=(key,controls,xLabel,yLabel,calculate)=>studies['complete-numerics-'+key]={controls,xLabel,yLabel,calculate};
  function solve(A,b) {
    const a=A.map((r,i)=>[...r,b[i]]),n=b.length;
    for(let j=0;j<n;j++) {
      let pivot=j;for(let i=j+1;i<n;i++)if(Math.abs(a[i][j])>Math.abs(a[pivot][j]))pivot=i;
      [a[j],a[pivot]]=[a[pivot],a[j]];
      if(Math.abs(a[j][j])<1e-25)throw Error('Singular browser matrix');
      for(let i=j+1;i<n;i++){const q=a[i][j]/a[j][j];for(let k=j;k<=n;k++)a[i][k]-=q*a[j][k];}
    }
    const x=Array(n);for(let i=n-1;i>=0;i--){let v=a[i][n];for(let j=i+1;j<n;j++)v-=a[i][j]*x[j];x[i]=v/a[i][i];}return x;
  }
  function eigen(A) {
    const a=A.map(r=>r.slice()),n=a.length;
    for(let step=0;step<80*n*n;step++){
      let p=0,q=1,big=0;
      for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)if(Math.abs(a[i][j])>big){big=Math.abs(a[i][j]);p=i;q=j;}
      if(big<1e-13)break;
      const theta=.5*Math.atan2(2*a[p][q],a[q][q]-a[p][p]),c=Math.cos(theta),s=Math.sin(theta);
      const app=a[p][p],aqq=a[q][q],apq=a[p][q];
      for(let k=0;k<n;k++)if(k!==p&&k!==q){const u=a[k][p],v=a[k][q];a[k][p]=a[p][k]=c*u-s*v;a[k][q]=a[q][k]=s*u+c*v;}
      a[p][p]=c*c*app-2*s*c*apq+s*s*aqq;a[q][q]=s*s*app+2*s*c*apq+c*c*aqq;a[p][q]=a[q][p]=0;
    }
    return a.map((r,i)=>r[i]).sort((a,b)=>a-b);
  }
  function gauss(n) {
    const nodes=[],weights=[];
    for(let i=0;i<n;i++){
      let x=Math.cos(Math.PI*(i+.75)/(n+.5)),d;
      for(let k=0;k<30;k++){let p=1,q=x;for(let j=2;j<=n;j++){const z=((2*j-1)*x*q-(j-1)*p)/j;p=q;q=z;}d=n*(x*q-p)/(x*x-1);const step=q/d;x-=step;if(Math.abs(step)<1e-15)break;}
      nodes.push(x);weights.push(2/((1-x*x)*d*d));
    }return {nodes,weights};
  }
  function interpolate(nodes,values) {
    const weights=nodes.map((x,i)=>1/nodes.reduce((p,y,j)=>i===j?p:p*(x-y),1));
    return x=>{let a=0,b=0;for(let i=0;i<nodes.length;i++){if(Math.abs(x-nodes[i])<1e-14)return values[i];const w=weights[i]/(x-nodes[i]);a+=w*values[i];b+=w;}return a/b;};
  }

  add('residual-direction',[['Weak singular value s',.01,1,.01,.02],['Selected residual angle (degrees)',0,90,1,45]],'Residual angle (degrees)','Error / residual norm',([s,a])=>{
    const gain=t=>Math.hypot(Math.cos(t*Math.PI/180),Math.sin(t*Math.PI/180)/s);
    return {curves:[curve('Directional amplification',xs(0,90),gain),curve('Condition-number bound',xs(0,90),()=>1/s),{name:'Selected direction',points:[[a,gain(a)]]}],result:'Selected amplification '+fmt(gain(a))+'. All residuals have norm one; κ₂(A) = '+fmt(1/s)+'.'};
  });
  add('pivot-growth',[['Largest dimension',2,30,1,12],['Last-column scale',.5,2,.1,1]],'Matrix dimension','log₂ element growth',([n,scale])=>{
    function growth(k){const a=Array.from({length:k},(_,i)=>Array.from({length:k},(_,j)=>j===k-1?scale:i===j?1:i>j?-1:0));let peak=Math.max(1,scale);for(let j=0;j<k-1;j++)for(let i=j+1;i<k;i++){const f=a[i][j]/a[j][j];for(let l=j;l<k;l++){a[i][l]-=f*a[j][l];peak=Math.max(peak,Math.abs(a[i][l]));}}return peak/Math.max(1,scale);}
    return {curves:[curve('Tie-preserving partial pivoting',xs(2,n,Math.max(1,n-2)),k=>Math.log2(growth(Math.round(k)))),curve('Poisson tridiagonal family',xs(2,n,Math.max(1,n-2)),()=>0)],result:'At dimension '+n+', growth factor '+fmt(growth(n))+'. This is intermediate element growth, not an observed forward-error factor.'};
  });
  add('line-prediction',[['Calibration half-width',.1,2,.1,1],['Independent sensor SD',.05,1,.05,.2]],'Prediction location x','Standard deviation',([a,sigma])=>{
    const locations=xs(-a,a,5),sum=dot(locations,locations),sd=x=>sigma*Math.sqrt(1/6+x*x/sum);
    return {curves:[curve('Uncertainty of fitted mean',xs(-3,3),sd),curve('Uncertainty of a new reading',xs(-3,3),x=>Math.hypot(sd(x),sigma))],result:'Slope SD '+fmt(sigma/Math.sqrt(sum))+'; intercept SD '+fmt(sigma/Math.sqrt(6))+'. Six equally spaced calibration readings have equal, independent variance.'};
  });
  add('cg-clusters',[['Extreme eigenvalue ratio',2,200,2,100],['Unknowns',6,30,2,20]],'CG iteration','log₁₀ relative energy error',([condition,n])=>{
    function run(cluster){const diag=xs(0,1,n-1).map((t,i)=>cluster?(i<n/2?1:condition):1+(condition-1)*t);let x=Array(n).fill(0),r=Array(n).fill(1),p=r.slice(),rr=dot(r,r);const base=Math.sqrt(diag.reduce((s,a)=>s+1/a,0)),points=[[0,0]];for(let k=1;k<=n;k++){const ap=p.map((v,i)=>v*diag[i]),den=dot(p,ap);if(den<1e-28)break;const alpha=rr/den;x=x.map((v,i)=>v+alpha*p[i]);r=r.map((v,i)=>v-alpha*ap[i]);const next=dot(r,r),err=Math.sqrt(x.reduce((s,v,i)=>s+diag[i]*(v-1/diag[i])**2,0))/base;points.push([k,log(err)]);if(next<1e-28)break;p=r.map((v,i)=>v+next/rr*p[i]);rr=next;}return points;}
    return {curves:[{name:'Two distinct eigenvalues',points:run(true)},{name:'Spread-out spectrum',points:run(false)}],result:'Both matrices have condition number '+condition+'. Exact arithmetic needs at most two CG steps for the two-eigenvalue case.'};
  });
  add('jacobi-smoothing',[['Mesh intervals',8,128,8,32],['Relaxation weight ω',.1,1.2,.05,.65],['Sweeps',1,20,1,1]],'Sine mode number','Signed mode amplitude',([n,w,k])=>({curves:[curve('Unweighted Jacobi',xs(1,n-1,n-2),j=>Math.cos(j*Math.PI/n)**k),curve('Weighted Jacobi',xs(1,n-1,n-2),j=>(1-w+w*Math.cos(j*Math.PI/n))**k)],result:'Slowest weighted one-step factor '+fmt(1-w+w*Math.cos(Math.PI/n))+'. After '+k+' sweeps the signs distinguish phase reversal from attenuation.'}));
  add('qr-gap',[['Smaller eigenvalue ratio ρ',.1,.99,.01,.8],['Initial rotation (degrees)',5,85,5,35]],'Unshifted QR iteration','log₁₀ off-diagonal / spectral gap',([rho,angle])=>{
    const t=Math.tan(angle*Math.PI/180),points=xs(0,80,80).map(k=>{const z=t*rho**k;return[k,log(Math.abs(z/(1+z*z)))];});return {curves:[{name:'Orthogonal-similarity mixing',points}],result:'Asymptotic contraction factor '+fmt(rho)+'. The eigenvalues remain 1 and '+fmt(rho)+' throughout the iteration.'};
  });
  add('rank-tail',[['Singular directions',4,40,2,20],['Geometric decay ratio',.1,1,.05,.7]],'Retained rank','Relative Frobenius error',([n,r])=>{
    const spectrum=Array.from({length:n},(_,i)=>r**i),sum=dot(spectrum,spectrum);
    return {curves:[curve('Geometric spectrum',xs(0,n,n),k=>Math.sqrt(spectrum.slice(k).reduce((s,v)=>s+v*v,0)/sum)),curve('Flat spectrum',xs(0,n,n),k=>Math.sqrt((n-k)/n))],result:'The best spectral-norm error after rank k is σk+1; this plot instead measures the complete discarded energy.'};
  });
  add('schur-star',[['First terminal conductance (S)',.1,5,.1,1],['Second terminal conductance (S)',.1,5,.1,2],['Selected third conductance (S)',0,8,.1,3]],'Third terminal conductance (S)','Equivalent branch conductance (S)',([a,b,c])=>({curves:[curve('Terminal 1–2',xs(0,8),g=>a*b/(a+b+g)),curve('Terminal 1–3',xs(0,8),g=>a*g/(a+b+g)),curve('Terminal 2–3',xs(0,8),g=>b*g/(a+b+g))],result:'At g₃ = '+fmt(c)+' S the eliminated 1–2 branch has conductance '+fmt(a*b/(a+b+c))+' S. All three couplings come from the same Schur complement.'}));
  add('edge-leverage',[['Direct AB conductance (S)',.05,10,.05,1],['Bridge CD conductance (S)',.1,5,.1,1]],'Direct AB conductance (S)','Dimensionless resistance R/(1 Ω) or leverage',([g,bridge])=>({curves:[curve('AB resistance / 1 Ω',xs(.05,10),v=>1/(v+.5)),curve('AB leverage',xs(.05,10),v=>v/(v+.5)),curve('Bridge leverage',xs(.05,10),()=>1)],result:'Resistance is normalised by 1 Ω on this shared dimensionless axis. Selected AB resistance '+fmt(1/(g+.5))+' Ω, leverage '+fmt(g/(g+.5))+'. Bridge resistance '+fmt(1/bridge)+' Ω and bridge leverage 1; the four edge leverages sum to 3.'}));
  add('sturm-count',[['Beads',4,40,2,12],['Central mass / other masses',.25,4,.25,2],['Selected squared frequency',.1,6,.1,2]],'Squared frequency shift (k/m units)','Number of eigenvalues below the shift',([n,mass,selected])=>{
    const masses=Array(n).fill(1);masses[Math.floor(n/2)]=mass;
    function count(s){let negative=0,d=0;for(let i=0;i<n;i++){d=2/masses[i]-s-(i?1/(masses[i]*masses[i-1]*d):0);if(Math.abs(d)<1e-14)d=-1e-14;if(d<0)negative++;}return negative;}
    return {curves:[curve('LDLᵀ sign count',xs(0,6,300),count),{name:'Selected shift',points:[[selected,count(selected)]]}],result:'There are '+count(selected)+' modes below the selected shift. Near a zero pivot this illustration uses a small negative pivot, corresponding to the upper-side limit of the counting step.'};
  });
  add('lanczos',[['Separated largest eigenvalue',5,30,1,12],['Starting overlap with highest mode',0,1,.05,.4],['Krylov dimension',1,6,1,4]],'Krylov dimension','Ritz value',([top,overlap,steps])=>{
    const diag=[1,1.4,2,2.6,3.4,top],v=[1,1,1,1,1,overlap],basis=[v.map(x=>x/norm(v))],alphas=[],betas=[],curves=diag.map((_,i)=>({name:'Ritz branch '+(i+1),points:[]}));
    for(let k=0;k<steps;k++){const q=basis[k],aq=q.map((x,i)=>diag[i]*x),alpha=dot(q,aq);alphas.push(alpha);let w=aq.map((x,i)=>x-alpha*q[i]-(k?betas[k-1]*basis[k-1][i]:0));for(const old of basis){const projection=dot(old,w);w=w.map((x,i)=>x-projection*old[i]);}const beta=norm(w),T=alphas.map((a,i)=>alphas.map((_,j)=>i===j?a:Math.abs(i-j)===1?betas[Math.min(i,j)]:0));eigen(T).forEach((value,i)=>curves[i].points.push([k+1,value]));if(beta<1e-12)break;betas.push(beta);basis.push(w.map(x=>x/beta));}
    return {curves:curves.filter(c=>c.points.length),result:'Exact spectrum: '+diag.map(fmt).join(', ')+'. Full reorthogonalisation suppresses duplicate directions. Zero initial highest-mode overlap makes that eigenvalue invisible.'};
  });

  add('bisection',[['Positive cubic target a',.1,8,.1,2],['Bisection updates',1,35,1,12]],'Bisection update','log₁₀ location error / certificate',([a,n])=>{
    const root=Math.cbrt(a);let lo=0,hi=3;const actual=[],bound=[];for(let k=0;k<=n;k++){const mid=(lo+hi)/2;actual.push([k,log(mid-root)]);bound.push([k,log((hi-lo)/2)]);if(mid**3<a)lo=mid;else hi=mid;}
    return {curves:[{name:'Actual midpoint error',points:actual},{name:'Certified half-width',points:bound}],result:'Root reference '+fmt(root)+'. The bracket [0, 3] encloses every offered target; a lucky midpoint can have a much smaller error than the guaranteed half-width.'};
  });
  add('relaxation',[['Small Jacobian eigenvalue',.1,1,.1,.5],['Large Jacobian eigenvalue',1,5,.1,2],['Chosen relaxation ω',.05,2,.05,.8]],'Relaxation parameter ω','Absolute error amplification',([a,b,w])=>({curves:[curve('Slow mode',xs(0,2),x=>Math.abs(1-x*a)),curve('Fast mode',xs(0,2),x=>Math.abs(1-x*b))],result:'At chosen ω the worst factor is '+fmt(Math.max(Math.abs(1-w*a),Math.abs(1-w*b)))+'. The minimax choice is '+fmt(2/(a+b))+' with factor '+fmt((b-a)/(a+b))+'.'}));
  add('newton-secant',[['Positive target a',.2,8,.1,2],['Initial Newton value',.2,4,.1,1],['Iterations',2,12,1,7]],'Iteration','log₁₀ root error',([a,start,n])=>{
    const root=Math.sqrt(a),newton=[],secant=[];let x=start,u=start,v=start+.2;
    for(let k=0;k<=n;k++){newton.push([k,log(x-root)]);secant.push([k,log(v-root)]);x=(x+a/x)/2;const next=(u+v)!==0?(u*v+a)/(u+v):v;u=v;v=next;}
    return {curves:[{name:'Newton for x² − a',points:newton},{name:'Secant for x² − a',points:secant}],result:'Positive reference root '+fmt(root)+'. Secant starts at x₀ and x₀ + 0.2; both methods recompute their local slopes at each step.'};
  });
  add('arctan-search',[['Initial point',.2,5,.1,2],['Accepted step fraction',.05,1,.05,.5]],'Fraction of Newton step','Half squared residual',([x,alpha])=>{
    const f=Math.atan(x),step=-(1+x*x)*f,merit=t=>.5*Math.atan(x+t*step)**2;
    return {curves:[curve('Actual merit',xs(0,1),merit),curve('Linear residual prediction',xs(0,1),t=>.5*f*f*(1-t)**2),{name:'Chosen step',points:[[alpha,merit(alpha)]]}],result:'Full Newton step '+fmt(step)+'. Chosen fraction changes merit by '+fmt(merit(alpha)-merit(0))+'. Negative change is improvement.'};
  });
  add('root-perturbation',[['Maximum log₁₀ coefficient perturbation',-8,-1,1,-3],['Repeated-root multiplicity',2,6,1,2]],'log₁₀ coefficient perturbation δ','log₁₀ displacement from root 1',([end,m])=>({curves:[curve('Simple root of x² − 1 − δ',xs(-14,end),p=>log(10**p/(Math.sqrt(1+10**p)+1))),curve('Root of (x − 1)ᵐ − δ',xs(-14,end),p=>p/m)],result:'Multiplicity '+m+' gives displacement δ^(1/'+m+'). A polynomial residual and a root-location certificate answer different questions.'}));
  add('interval-newton',[['Centre c',1.42,2,.01,1.5],['Selected initial half-width',.05,.4,.01,.25]],'Initial half-width','Root or interval endpoint',([c,h])=>{
    const endpoints=w=>{const f=c*c-2,a=f/(2*(c-w)),b=f/(2*(c+w));return [Math.max(c-w,c-Math.max(a,b)),Math.min(c+w,c-Math.min(a,b))];};const [lo,hi]=endpoints(h);
    return {curves:[curve('Intersected lower endpoint',xs(.05,.4),w=>endpoints(w)[0]),curve('Intersected upper endpoint',xs(.05,.4),w=>endpoints(w)[1]),curve('Reference √2',xs(.05,.4),()=>Math.SQRT2)],result:lo<=hi?'Selected intersection ['+fmt(lo)+', '+fmt(hi)+']. Ordinary browser arithmetic illustrates the construction; outward rounding is required for a machine certificate.':'The selected intersection is empty: this interval excludes the positive root. Ordinary browser arithmetic is used.'};
  });
  add('contact',[['Spring stiffness (N/m)',.5,4,.1,1],['Stop displacement (m)',.25,2,.05,1]],'Applied force (N)','Displacement / gap or reaction / (k × gap)',([k,gap])=>({curves:[curve('Normalised spring displacement',xs(0,6),f=>Math.min(f/k,gap)/gap),curve('Normalised contact reaction',xs(0,6),f=>Math.max(0,f-k*gap)/(k*gap))],result:'Contact engages at applied force '+fmt(k*gap)+' N. Both displayed outputs are dimensionless: displacement divided by the stop gap, and reaction divided by k times that gap.'}));
  add('newton-basin',[['Real initial value',-1.5,1.5,.05,.1],['Imaginary initial value',-1.5,1.5,.05,0],['Maximum iterations',4,60,1,30]],'Newton step','log₁₀ cubic residual',([re,im,n])=>{
    const points=[];let x=re,y=im,status='Iteration limit reached';for(let k=0;k<=n;k++){const residual=Math.hypot(x*x*x-3*x*y*y-1,3*x*x*y-y*y*y);points.push([k,log(residual)]);if(residual<1e-10){status='Converged to '+fmt(x)+' + '+fmt(y)+' i';break;}const square=x*x+y*y;if(square<1e-24){status='Derivative vanishes at zero; no Newton step exists';break;}const u=(x*x-y*y)/(square*square),v=-2*x*y/(square*square);x=(2*x+u)/3;y=(2*y+v)/3;}
    return {curves:[{name:'Chosen complex starting point',points}],result:status+'. Move either coordinate slightly near a basin boundary to test sensitivity.'};
  });
  add('covariance-root',[['Target eigenvalue ratio',1,16,1,9],['Starting orientation (degrees)',0,90,5,45]],'Newton iteration','log₁₀ Frobenius residual',([ratio,angle])=>{
    const t=angle*Math.PI/180,c=Math.cos(t),s=Math.sin(t);let X=[[c*c+2*s*s,-c*s],[-c*s,s*s+2*c*c]];const target=[[1,0],[0,ratio]],history=[];
    for(let k=0;k<8;k++){const square=X.map(r=>[dot(r,[X[0][0],X[1][0]]),dot(r,[X[0][1],X[1][1]])]);const R=target.map((r,i)=>r.map((v,j)=>v-square[i][j]));history.push([k,log(Math.hypot(...R.flat()))]);const a=X[0][0],b=X[0][1],d=X[1][1],h=solve([[2*a,2*b,0],[b,a+d,b],[0,2*b,2*d]],[R[0][0],R[0][1],R[1][1]]);X=[[a+h[0],b+h[1]],[b+h[1],d+h[2]]];}
    return {curves:[{name:'Symmetric Sylvester Newton correction',points:history}],result:'Target positive square root is diag(1, '+fmt(Math.sqrt(ratio))+'). Each step solves XH + HX = A − X² in the three symmetric matrix coordinates.'};
  });

  add('local-interpolation',[['Interval length',.2,2,.1,1],['Exponential growth rate',.2,3,.1,1]],'Position x','Interpolation minus exact value',([length,rate])=>{
    const f=x=>Math.exp(rate*x),line=interpolate([0,length],[1,f(length)]),quad=interpolate([0,length/2,length],[1,f(length/2),f(length)]);
    return {curves:[curve('Two endpoint values',xs(0,length),x=>line(x)-f(x)),curve('Endpoints and midpoint',xs(0,length),x=>quad(x)-f(x))],result:'All nodal errors vanish. Halve the interval length at fixed rate to compare the interior error of linear and quadratic interpolation.'};
  });
  add('hermite-data',[['Initial slope',0,4,.1,2],['Final observed value',2,6,.1,4]],'Coordinate x','Interpolated value',([slope,end])=>({curves:[curve('Value-and-slope quadratic',xs(0,1),x=>1+slope*x+(end-1-slope)*x*x),curve('Endpoint-only line',xs(0,1),x=>1+(end-1)*x)],result:'Midpoint value '+fmt(1+slope/2+(end-1-slope)/4)+'. Changing the initial slope by Δ changes the interior by Δx(1 − x), with no endpoint change.'}));
  add('shape-spline',[['Small initial increments',.01,.09,.01,.04],['Final plateau',.2,2,.1,1]],'Sample time','Cumulative reading',([inc,top])=>{
    const y=[0,inc,2*inc,top,top],slopes=y.slice(1).map((v,i)=>v-y[i]);
    const second=[0,...solve([[4,1,0],[1,4,1],[0,1,4]],slopes.slice(1).map((v,i)=>6*(v-slopes[i]))),0];
    const p=[Math.max(0,(3*slopes[0]-slopes[1])/2),...slopes.slice(1).map((v,i)=>v*slopes[i]>0?2*v*slopes[i]/(v+slopes[i]):0),0];
    const natural=x=>{const j=Math.min(3,Math.floor(x)),t=x-j;return (1-t)*y[j]+t*y[j+1]+(((1-t)**3-(1-t))*second[j]+(t**3-t)*second[j+1])/6;};
    const monotone=x=>{const j=Math.min(3,Math.floor(x)),t=x-j;return (2*t**3-3*t*t+1)*y[j]+(t**3-2*t*t+t)*p[j]+(-2*t**3+3*t*t)*y[j+1]+(t**3-t*t)*p[j+1];};
    return {curves:[curve('Natural cubic spline',xs(0,4,200),natural),curve('Shape-preserving Hermite',xs(0,4,200),monotone),{name:'Observations',points:y.map((v,i)=>[i,v])}],result:'Both curves interpolate the same five monotone observations. Inspect the final plateau for an overshoot and a subsequent negative slope.'};
  });
  add('composite-quadrature',[['Exponential rate',.2,5,.2,1],['Maximum panels',4,80,4,32]],'Number of integration panels','log₁₀ absolute integration error',([rate,n])=>{
    const exact=Math.expm1(rate)/rate,counts=xs(2,n,n/2-1);function value(N,simpson){let sum=1+Math.exp(rate);for(let j=1;j<N;j++)sum+=(simpson?(j%2?4:2):2)*Math.exp(rate*j/N);return sum/(N*(simpson?3:2));}
    return {curves:[curve('Composite trapezoid',counts,N=>log(value(N,false)-exact)),curve('Composite Simpson',counts,N=>log(value(N,true)-exact))],result:'Exact integral '+fmt(exact)+'. The even panel counts allow both methods on the same nodes; compare second and fourth order under refinement.'};
  });
  add('gauss-peak',[['Peak half-width a',.02,.5,.01,.1],['Highest Gauss order',4,32,1,16]],'Gauss–Legendre nodes','log₁₀ absolute integration error',([a,n])=>{
    const exact=2*Math.atan(1/a)/a;return {curves:[curve('Unsplit interval [−1, 1]',xs(1,n,n-1),N=>{const q=gauss(N);return log(q.nodes.reduce((s,x,i)=>s+q.weights[i]/(x*x+a*a),0)-exact);})],result:'Exact integral '+fmt(exact)+'. Odd orders hit the peak centre; even orders straddle it. Decrease a before increasing the node budget.'};
  });
  add('adaptive-alias',[['Oscillations of sin² across interval',1,12,1,4],['Phase offset (radians)',0,1.5,.05,0]],'Position x','Integrand / sampled value',([f,phase])=>{
    const fn=x=>Math.sin(Math.PI*f*x+phase)**2,nodes=xs(0,1,4),coarse=(fn(0)+4*fn(.5)+fn(1))/6,fine=(fn(0)+4*fn(.25)+2*fn(.5)+4*fn(.75)+fn(1))/12;
    return {curves:[curve('Continuous integrand',xs(0,1,480),fn),{name:'Nested Simpson samples',points:nodes.map(x=>[x,fn(x)])}],result:'Exact integral 1/2. Coarse Simpson '+fmt(coarse)+', refined Simpson '+fmt(fine)+', difference-based estimate '+fmt(Math.abs(fine-coarse)/15)+'. Agreement can be false convergence when every node misses the oscillation.'};
  });
  add('fourier-alias',[['Samples per second',4,24,1,8],['Base frequency (Hz)',1,8,1,2]],'Time (s)','Signal',([rate,f])=>({curves:[curve('Base sinusoid',xs(0,1,600),t=>Math.sin(2*Math.PI*f*t)),curve('Aliased frequency',xs(0,1,600),t=>Math.sin(2*Math.PI*(f+rate)*t)),{name:'Shared recorded values',points:xs(0,1,rate).map(t=>[t,Math.sin(2*Math.PI*f*t)])}],result:'Frequencies '+f+' and '+(f+rate)+' Hz agree at every sample j/'+rate+'. Adding display points between these readings does not recover the lost band information.'}));
  add('stratified-rms',[['Integrand exponent p',1,4,.25,1],['Maximum sample count',16,128,16,64]],'Independent function evaluations','RMS integration error',([p,n])=>{
    const direct=1/(2*p+1)-1/(p+1)**2;function strat(N){let variance=0;for(let i=0;i<N;i++){const a=i/N,b=(i+1)/N,mean=N*(b**(p+1)-a**(p+1))/(p+1),second=N*(b**(2*p+1)-a**(2*p+1))/(2*p+1);variance+=Math.max(0,second-mean*mean)/N**2;}return Math.sqrt(variance);}
    return {curves:[curve('Independent uniform samples',xs(2,n,n-2),N=>Math.sqrt(direct/N)),curve('One independent sample per stratum',xs(2,n,n-2),strat)],result:'For x^'+fmt(p)+', exact integral '+fmt(1/(p+1))+'. These are ensemble RMS errors computed from exact within-cell moments, not one random run.'};
  });
  add('tanh-sinh',[['Endpoint exponent α',.1,.95,.05,.5],['Truncation |t| ≤ T',1,4,.25,3],['Finest transformed panels',16,160,16,64]],'Transformed trapezoid panels','log₁₀ absolute integration error',([alpha,T,n])=>{
    const exact=1/(1-alpha),integrand=t=>{const z=Math.PI*Math.sinh(t),logX=z<0?z-Math.log1p(Math.exp(z)):-Math.log1p(Math.exp(-z)),logOneMinus=z>0?-z-Math.log1p(Math.exp(-z)):-Math.log1p(Math.exp(z));return Math.PI*Math.cosh(t)*Math.exp((1-alpha)*logX+logOneMinus);};
    const counts=xs(8,n,(n-8)/8);function integrate(N){let sum=.5*(integrand(-T)+integrand(T));for(let i=1;i<N;i++)sum+=integrand(-T+2*T*i/N);return sum*2*T/N;}
    const lo=1/(1+Math.exp(Math.PI*Math.sinh(T))),hi=1-lo,tail=(lo**(1-alpha)+1-hi**(1-alpha))/(1-alpha);
    return {curves:[curve('Tanh–sinh truncation plus sampling error',counts,N=>log(integrate(N)-exact)),curve('Exact omitted-tail contribution',counts,()=>log(tail))],result:'Exact integral '+fmt(exact)+'; omitted tails contribute '+fmt(tail)+'. Increasing panel count cannot remove a fixed truncation error. The transformed integrand uses logarithms to retain tiny endpoint weights.'};
  });
  add('slope-window',[['Noise standard deviation',0,.1,.005,.02],['Cubic coefficient',0,3,.1,1],['Selected sample spacing',.02,.5,.02,.2]],'Sample spacing h','Root mean square slope error',([sigma,cubic,h])=>{
    function stats(radius,s){let j2=0,j4=0;for(let j=-radius;j<=radius;j++){j2+=j*j;j4+=j**4;}const bias=cubic*s*s*j4/j2,sd=sigma/(s*Math.sqrt(j2));return Math.hypot(bias,sd);}
    return {curves:[curve('Three-point quadratic fit',xs(.01,.6),s=>stats(1,s)),curve('Five-point quadratic fit',xs(.01,.6),s=>stats(2,s))],result:'At h = '+fmt(h)+', slope RMSE is '+fmt(stats(1,h))+' for three samples and '+fmt(stats(2,h))+' for five. The wider fit reduces noise but admits more cubic bias.'};
  });
  add('regularity',[['Largest even Chebyshev degree',4,40,2,20],['Corner exponent p',.25,2,.25,1]],'Polynomial degree','log₁₀ sampled maximum interpolation error',([n,p])=>{
    const test=xs(-1,1,401),funcs=[x=>Math.exp(x),x=>Math.abs(x)**p,x=>x<.07?-1:1];
    return {curves:funcs.map((f,j)=>({name:['Analytic exp(x)','Corner |x|ᵖ','Jump at x = 0.07'][j],points:xs(2,n,n/2-1).map(N=>{const nodes=xs(0,Math.PI,N).map(Math.cos),poly=interpolate(nodes,nodes.map(f));return[N,log(Math.max(...test.map(x=>Math.abs(poly(x)-f(x)))))];})})),result:'Errors are sampled on 402 fixed test locations, not certified supremum bounds. Smoothness changes the attainable rate; the jump remains incompatible with uniform convergence by continuous polynomials.'};
  });
  add('rational-pole',[['Pole location a outside interval',1.2,5,.1,2],['Fitted pole error',-.15,.5,.05,0]],'Coordinate x on [−1, 1]','Approximation error',([a,bias])=>{
    const slope=1/(a*a-1),extreme=a-Math.sqrt(a*a-1),f=x=>1/(a-x),chord=x=>(a+x)/(a*a-1),amplitude=(chord(extreme)-f(extreme))/2;
    return {curves:[curve('Best affine approximation',xs(-1,1),x=>chord(x)-amplitude-f(x)),curve('Rational with chosen pole',xs(-1,1),x=>1/(a+bias-x)-f(x))],result:'Best affine error magnitude '+fmt(amplitude)+' with interior extremum x = '+fmt(extreme)+'. A correctly placed single pole reproduces this target exactly; pole error introduces a structured bias.'};
  });

  add('defect-growth',[['Growth rate λ',-2,1,.1,.6],['Defect rate',.1,2,.1,1],['End time',1,8,.5,4]],'Time','Accumulated state error',([rate,defect,end])=>({curves:[curve('Chosen propagator',xs(0,end),t=>defect*(Math.abs(rate)<1e-12?t:Math.expm1(rate*t)/rate)),curve('Neutral propagator',xs(0,end),t=>defect*t)],result:'This solves e′ = λe + defect from e(0) = 0. Equal local defects can saturate, accumulate linearly or grow exponentially depending on the underlying evolution.'}));
  add('stiff-amplification',[['Maximum step × rate',1,30,1,12],['Selected step × rate',.1,20,.1,10]],'Step × positive decay rate','Signed one-step amplification',([end,r])=>({curves:[curve('Exact exponential',xs(0,end),x=>Math.exp(-x)),curve('Backward Euler',xs(0,end),x=>1/(1+x)),curve('Trapezoidal',xs(0,end),x=>(1-x/2)/(1+x/2))],result:'At selected hk = '+fmt(r)+': exact '+fmt(Math.exp(-r))+', backward Euler '+fmt(1/(1+r))+', trapezoidal '+fmt((1-r/2)/(1+r/2))+'. Negative amplification creates an alternating unresolved mode.'}));
  add('step-controller',[['Estimator power p',1,6,1,3],['Safety factor',.5,1,.05,.9],['Selected error / tolerance',.1,10,.1,8]],'Current error / tolerance','Proposed next step / current step',([p,safety,ratio])=>({curves:[curve('Power-law step proposal',xs(.1,10),r=>safety*r**(-1/p)),curve('No step change',xs(.1,10),()=>1)],result:'Selected ratio requests hnext/h = '+fmt(safety*ratio**(-1/p))+'. Accept/reject the current step separately; this rule assumes the estimator scales as h^'+p+'.'}));
  add('splitting-order',[['Oscillator frequency ω',.5,4,.5,1],['Maximum steps to t = 1',8,80,4,32]],'Steps to time 1','log₁₀ Euclidean state error',([omega,n])=>{
    function run(N,symmetric){let q=1,p=0;const h=1/N;for(let j=0;j<N;j++){if(symmetric){q+=h*p/2;p-=h*omega*omega*q;q+=h*p/2;}else{q+=h*p;p-=h*omega*omega*q;}}return Math.hypot(q-Math.cos(omega),p+omega*Math.sin(omega));}
    return {curves:[curve('Lie composition',xs(4,n,(n-4)/4),N=>log(run(N,false))),curve('Symmetric Strang composition',xs(4,n,(n-4)/4),N=>log(run(N,true)))],result:'Each shear substep is exact. The error comes from composition of noncommuting flows; expect first versus second global order.'};
  });
  add('verlet-phase',[['Selected hω',.05,1.95,.05,1],['Physical cycles',1,40,1,10]],'Step × physical frequency','Numerical / physical frequency',([r,cycles])=>{
    const ratio=x=>2*Math.asin(x/2)/x;
    return {curves:[curve('Velocity Verlet',xs(.01,1.95),ratio),curve('Exact frequency',xs(.01,1.95),()=>1)],result:'Selected frequency ratio '+fmt(ratio(r))+'; phase lead after '+cycles+' exact cycles is '+fmt(2*Math.PI*cycles*(ratio(r)-1))+' rad. The stability boundary is hω = 2.'};
  });
  add('exponential-forcing',[['Decay rate k (1/s)',1,100,1,100],['Input ramp slope',0,30,1,10],['Steps to time 1',4,80,4,20]],'Time (s)','State',([k,ramp,n])=>{
    const h=1/n,base=.2*k,exact=t=>.2+.8*Math.exp(-k*t)+ramp*(t/k+Math.expm1(-k*t)/(k*k));let frozen=1,implicit=1;const a=[[0,1]],b=[[0,1]];
    for(let j=0;j<n;j++){frozen=Math.exp(-k*h)*frozen-Math.expm1(-k*h)*(base+ramp*j*h)/k;implicit=(implicit+h*(base+ramp*(j+1)*h))/(1+k*h);a.push([(j+1)*h,frozen]);b.push([(j+1)*h,implicit]);}
    return {curves:[curve('Exact ramp response',xs(0,1,250),exact),{name:'Exponential Euler, frozen input',points:a},{name:'Backward Euler',points:b}],result:'Frozen-input endpoint error '+fmt(Math.abs(frozen-exact(1)))+'. With zero ramp, exponential Euler reproduces the complete solution at its time nodes.'};
  });
  add('rattle-clock',[['Step × angular speed',.05,.95,.05,.5],['Steps',4,80,4,20]],'Time × angular speed','Numerical minus exact angle (rad)',([h,n])=>({curves:[curve('RATTLE',xs(0,n,n),j=>j*(Math.asin(h)-h)),curve('Half the step, same elapsed time',xs(0,2*n,2*n),j=>j*(Math.asin(h/2)-h/2))].map(c=>({...c,points:c.points.map(([j,e])=>[j*(c.name==='RATTLE'?h:h/2),e])})),result:'The constrained radius and speed are exact, but each full step advances angle asin(hω). The full-step terminal phase lead is '+fmt(n*(Math.asin(h)-h))+' rad; no real constraint update exists when hω > 1.'}));
  add('euler-extrapolation',[['Growth rate λ',-.8,2,.2,1],['Extrapolation columns',1,6,1,4]],'Cumulative Euler slope evaluations','log₁₀ endpoint error at t = 1',([rate,columns])=>{
    const exact=Math.exp(rate),rows=[],curves=Array.from({length:columns},(_,j)=>({name:'Column '+(j+1),points:[]}));let cost=0;
    for(let k=0;k<10;k++){const n=k+1;cost+=n;rows[k]=[(1+rate/n)**n];for(let j=1;j<=Math.min(k,columns-1);j++)rows[k][j]=rows[k][j-1]+(rows[k][j-1]-rows[k-1][j-1])/(n/(n-j)-1);for(let j=0;j<Math.min(k+1,columns);j++)curves[j].points.push([cost,log(rows[k][j]-exact)]);}
    return {curves,result:'The exact endpoint is exp(λ) = '+fmt(exact)+'. Each Euler subdivision has an expansion in 1/n; extrapolation cancels successive terms while reusing the computed table.'};
  });
  add('constraint-differences',[['Position disturbance order p',2,6,1,3],['Disturbance coefficient',.1,3,.1,1]],'log₁₀ time step h','log₁₀ disturbance magnitude',([p,a])=>({curves:[curve('Position a hᵖ',xs(-4,-.5),x=>Math.log10(a)+p*x),curve('First difference / h',xs(-4,-.5),x=>Math.log10(2*a)+(p-1)*x),curve('Second difference / h²',xs(-4,-.5),x=>Math.log10(4*a)+(p-2)*x)],result:'An alternating disturbance loses one power of h under each discrete derivative. For p = '+p+', the induced multiplier contribution scales as h^'+(p-2)+'.'}));

  add('flux-network',[['Middle conductance (W/K)',.2,8,.2,2],['Source at first interior node (W)',0,20,1,7],['Right boundary temperature (K)',0,20,1,10]],'Node along three-link chain','Temperature (K)',([g,source,right])=>{
    const values=solve([[1+g,-g],[-g,g+4]],[source,4*right]),base=solve([[1+g,-g],[-g,g+4]],[0,4*right]);
    return {curves:[{name:'With interior source',points:[[0,0],[1,values[0]],[2,values[1]],[3,right]]},{name:'Without interior source',points:[[0,0],[1,base[0]],[2,base[1]],[3,right]]}],result:'Additional boundary outflow '+fmt(values[0]-base[0]+4*(values[1]-base[1]))+' W equals the inserted '+source+' W. Both interior temperature increments are nonnegative.'};
  });
  add('point-load',[['Source strength Q',.2,5,.2,1],['Selected element coordinate a/h',0,1,.05,.3]],'Source position a/h within element','Nodal load',([Q,a])=>({curves:[curve('Left hat-function load',xs(0,1),x=>Q*(1-x)),curve('Right hat-function load',xs(0,1),x=>Q*x),curve('Nearest-node left load',xs(0,1),x=>x<.5?Q:0)],result:'Selected loads '+fmt(Q*(1-a))+' and '+fmt(Q*a)+' sum to Q = '+fmt(Q)+' and reproduce the source first moment. Nearest-node placement preserves only the total.'}));
  add('heat-ripple',[['Highest diffusion number',.5,3,.1,1.5],['Mode fraction of Nyquist',.1,1,.1,1]],'Diffusion number κΔt/Δx²','Signed one-step amplitude',([end,mode])=>{
    const eigenvalue=4*Math.sin(Math.PI*mode/2)**2;
    return {curves:[curve('Forward Euler',xs(0,end),r=>1-r*eigenvalue),curve('Crank–Nicolson',xs(0,end),r=>(1-r*eigenvalue/2)/(1+r*eigenvalue/2)),curve('Exact semidiscrete mode',xs(0,end),r=>Math.exp(-r*eigenvalue))],result:'Directional Laplacian magnitude times Δx² is '+fmt(eigenvalue)+'. Forward Euler is stable up to r = '+fmt(2/eigenvalue)+' and nonnegative only up to r = '+fmt(1/eigenvalue)+'.'};
  });
  add('transport-dispersion',[['Courant number ν',.05,1,.05,.5],['Propagation steps',1,20,1,1]],'Fourier angle θ (radians)','Amplitude after selected steps',([nu,n])=>{
    const values=theta=>[Math.hypot(1-nu+nu*Math.cos(theta),nu*Math.sin(theta)),Math.hypot(Math.cos(theta),nu*Math.sin(theta)),Math.hypot(1+nu*nu*(Math.cos(theta)-1),nu*Math.sin(theta))];
    return {curves:['Upwind','Lax–Friedrichs','Lax–Wendroff'].map((name,j)=>curve(name,xs(0,Math.PI,200),t=>values(t)[j]**n)),result:'Exact transport retains unit amplitude at every θ. All chosen Courant numbers lie in the stability range; grid-scale damping differs substantially between methods.'};
  });
  add('contact-resistance',[['First bulk resistance',.2,4,.2,1],['Second bulk resistance',.2,4,.2,2],['Selected contact resistance',0,8,.2,3]],'Contact resistance','Fraction of total temperature drop',([a,b,c])=>({curves:[curve('First bulk layer',xs(0,8),r=>a/(a+b+r)),curve('Contact jump',xs(0,8),r=>r/(a+b+r)),curve('Second bulk layer',xs(0,8),r=>b/(a+b+r))],result:'At selected contact resistance, the jump carries '+fmt(c/(a+b+c)*100)+'% of the imposed drop. The three fractions always sum to one.'}));
  add('advection-layer',[['Cell Péclet number',.2,5,.2,3],['Mesh cells',4,40,2,10]],'Position x','Scalar u, boundary values 0 and 1',([pe,n])=>{
    function discrete(upwind){const lower=-1-(upwind?pe:pe/2),upper=-1+(upwind?0:pe/2),diag=2+(upwind?pe:0),A=Array.from({length:n-1},(_,i)=>Array.from({length:n-1},(_,j)=>i===j?diag:j===i-1?lower:j===i+1?upper:0)),b=Array(n-1).fill(0);b[n-2]=-upper;return [0,...solve(A,b),1];}
    const central=discrete(false),upwind=discrete(true),total=pe*n;
    return {curves:[curve('Centred difference',xs(0,1,n),(_,i)=>central[i]),curve('Upwind difference',xs(0,1,n),(_,i)=>upwind[i]),curve('Continuum solution',xs(0,1,200),x=>(Math.exp(total*(x-1))-Math.exp(-total))/(1-Math.exp(-total)))],result:'Total Péclet number '+fmt(total)+'. The centred minimum is '+fmt(Math.min(...central))+'. The monotonicity threshold is cell Pe = 2.'};
  });
  add('green-output',[['Sensor coordinate a',.05,.95,.05,.25],['Point-source error strength',-.5,2,.1,1]],'Location of source error','Change in observed value',([a,Q])=>({curves:[curve('Adjoint Green sensitivity',xs(0,1,200),x=>Q*Math.min(x,a)*(1-Math.max(x,a)))],result:'Sensitivity peaks at the sensor: '+fmt(Q*a*(1-a))+'. Source errors at opposite signs can cancel in this output while their norm remains large.'}));
  add('mass-positivity',[['Interior nodes',3,15,2,3],['Diffusion number r',.02,.6,.02,.1]],'Position along unit rod','Temperature above zero boundary',([n,r])=>{
    const mass=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?4:Math.abs(i-j)===1?1:0)),rhs=Array.from({length:n},(_,i)=>i===0?12:i===1?-6:0),column=solve(mass,rhs),u0=Array.from({length:n},(_,i)=>i===0?1:0),consistent=u0.map((v,i)=>v-r*column[i]),lumped=u0.map((v,i)=>v-r*(i===0?2:i===1?-1:0)),x=xs(0,1,n+1),top=6*(1-Math.cos(n*Math.PI/(n+1)))/(2+Math.cos(n*Math.PI/(n+1)));
    return {curves:[curve('Consistent mass Euler step',x,(_,i)=>[0,...consistent,0][i]),curve('Lumped mass Euler step',x,(_,i)=>[0,...lumped,0][i])],result:'Consistent-mass stability limit r = '+fmt(2/top)+'. Third interior temperature '+fmt(consistent[2])+'. Spectral stability and preservation of nonnegative temperature are separate properties.'};
  });
  add('turing-modes',[['Diffusion Du',.01,.2,.01,.05],['Diffusion Dv',.2,4,.1,1],['Periodic grid cells',16,96,8,32]],'Integer mode on ring of length 2π','Largest real growth rate',([du,dv,n])=>{
    function growth(q){const trace=-1-(du+dv)*q,det=1-(dv-2*du)*q+du*dv*q*q;return (trace+Math.sqrt(Math.max(0,trace*trace-4*det)))/2;}
    return {curves:[curve('Continuum wavenumber',xs(0,n/2,n/2),m=>growth(m*m)),curve('Second-difference wavenumber',xs(0,n/2,n/2),m=>growth((n/Math.PI*Math.sin(Math.PI*m/n))**2))],result:'The spatially uniform mode has real growth −0.5. A positive plotted value identifies a diffusion-driven instability; refining the grid changes which wavelength the numerical model selects.'};
  });

  add('sensor-contrast',[['Independent repetitions',1,25,1,1],['Selected differential weight',.05,1,.05,.1]],'Differential weight a','Contrast standard error / one-reading σ',([n,a])=>({curves:[curve('Two-sensor contrast',xs(.05,1),x=>2*Math.sqrt(2/n)/x)],result:'Selected contrast standard error '+fmt(2*Math.sqrt(2/n)/a)+' times one-reading σ. Repetition improves as 1/√N; changing sensor geometry improves as 1/a.'}));
  add('calibration-floor',[['Shared calibration SD',0,1,.05,.3],['Independent reading SD',.1,2,.1,1]],'Repeated readings N','Standard deviation of mean',([common,independent])=>({curves:[curve('Correct common-offset covariance',xs(1,100,99),n=>Math.sqrt(common*common+independent*independent/n)),curve('Incorrect independent-offset assumption',xs(1,100,99),n=>Math.hypot(common,independent)/Math.sqrt(n))],result:'The limiting standard deviation is the shared calibration SD '+fmt(common)+'. Repeating the same instrument does not remove its common unknown offset.'}));
  add('bayesian-correlation',[['Measured first coordinate',-4,4,.2,3],['Measurement noise variance',.1,4,.1,1]],'Prior correlation ρ','Posterior mean / standard deviation',([datum,R])=>({curves:[curve('Unobserved-coordinate posterior mean',xs(-.95,.95),rho=>rho*datum/(1+R)),curve('Unobserved-coordinate posterior SD',xs(-.95,.95),rho=>Math.sqrt(1-rho*rho/(1+R)))],result:'The observed-coordinate mean is '+fmt(datum/(1+R))+' and its SD is '+fmt(Math.sqrt(R/(1+R)))+'. The second coordinate is updated only through the specified prior covariance.'}));
  add('penalty-shrinkage',[['Penalty weight λ',0,2,.1,.5]],'Measured scalar coefficient','Reconstructed coefficient',([lambda])=>({curves:[curve('Absolute-value penalty',xs(-3,3),y=>Math.sign(y)*Math.max(0,Math.abs(y)-lambda)),curve('Quadratic penalty λx²/2',xs(-3,3),y=>y/(1+lambda))],result:'The objectives are (x − y)²/2 + λ|x| and (x − y)²/2 + λx²/2. The first sets all |y| ≤ λ exactly to zero.'}));
  add('singular-noise',[['Weak singular value',.01,.5,.01,.05],['Noise in weak data coordinate',-.05,.05,.005,.02],['Regularisation μ',0,.3,.01,.05]],'Singular direction','True / reconstructed coordinate',([s,noise,mu])=>{
    const singular=[1,.3,s],truth=[1,.7,.4],data=truth.map((x,i)=>x*singular[i]+(i===2?noise:0)),raw=data.map((v,i)=>v/singular[i]),filtered=data.map((v,i)=>v*singular[i]/(singular[i]**2+mu**2));return {curves:[{name:'True coefficients',points:truth.map((v,i)=>[i+1,v])},{name:'Unregularised inverse',points:raw.map((v,i)=>[i+1,v])},{name:'Tikhonov inverse',points:filtered.map((v,i)=>[i+1,v])}],result:'Weak-mode noise is amplified by '+fmt(1/s)+' before regularisation. The regularised weak-mode resolution gain is '+fmt(s*s/(s*s+mu*mu))+'.'};
  });
  add('gauss-newton-fit',[['Initial rate (1/s)',0,2,.1,0],['Second observation offset',-.1,.1,.01,0],['Damping μ',0,2,.1,0]],'Iteration','Decay rate (1/s)',([start,offset,mu])=>{
    let rate=start;const points=[[0,rate]],data=[.5,.25+offset];for(let k=1;k<=10;k++){const prediction=[Math.exp(-rate),Math.exp(-2*rate)],j=[-prediction[0],-2*prediction[1]],residual=prediction.map((v,i)=>v-data[i]),step=-dot(j,residual)/(dot(j,j)+mu);let fraction=1;const old=dot(residual,residual);for(let back=0;back<18;back++){const trial=rate+fraction*step,errors=[Math.exp(-trial)-data[0],Math.exp(-2*trial)-data[1]];if(dot(errors,errors)<=old)break;fraction/=2;}rate+=fraction*step;points.push([k,rate]);}
    return {curves:[{name:'Damped Gauss–Newton with backtracking',points},curve('Unperturbed reference ln 2',xs(0,10,10),()=>Math.LN2)],result:'Final rate '+fmt(rate)+'. Changing the second observation makes the two data inconsistent with a single exact decay; the solver then seeks a least-squares compromise.'};
  });
  add('decay-design',[['Decay rate (1/s)',.2,2,.1,.8],['Observation noise SD',.02,.3,.02,.1],['Selected observation time (s)',.1,8,.1,1.2]],'Observation time (s)','Fisher information for decay rate',([k,sigma,t])=>{
    const info=x=>(x*Math.exp(-k*x)/sigma)**2;
    return {curves:[curve('One known-amplitude decay reading',xs(0,8),info)],result:'Maximum information occurs at t = 1/k = '+fmt(1/k)+' s. At the selected time the local rate SD is '+fmt(1/Math.sqrt(info(t)))+' 1/s. Unknown amplitude or correlated noise changes this design problem.'};
  });
  add('adjoint-gradient',[['Material parameter p',.1,3,.1,1],['Observed second state',.1,.8,.05,.4]],'log₁₀ parameter perturbation h','log₁₀ objective remainder',([p,y])=>{
    const objective=t=>.5*(1/(2*t+1)-y)**2,gradient=-2*(1/(2*p+1)-y)/(2*p+1)**2,base=objective(p);
    return {curves:[curve('Subtract correct adjoint gradient',xs(-8,-1),x=>{const h=10**x;return log(objective(p+h)-base-h*gradient);}),curve('Subtract wrong-sign gradient',xs(-8,-1),x=>{const h=10**x;return log(objective(p+h)-base+h*gradient);})],result:'Analytic and adjoint derivative = '+fmt(gradient)+'. A correct gradient leaves a quadratic remainder until cancellation dominates; at an exact fit this sign test loses sensitivity.'};
  });
  add('cgls-stop',[['Noise amplitude',0,.05,.0025,.01],['Singular-value decay power',.5,3,.25,1.5],['Displayed iterations',3,25,1,15]],'CGLS iteration','log₁₀ state error / data residual',([noise,power,steps])=>{
    const n=20,sigma=Array.from({length:n},(_,i)=>(i+1)**(-power)),truth=Array.from({length:n},(_,i)=>Math.exp(-i/4)),b=truth.map((v,i)=>sigma[i]*v+noise*Math.sin((i+1)*2.3));let x=Array(n).fill(0),r=b.slice(),s=r.map((v,i)=>v*sigma[i]),direction=s.slice(),gamma=dot(s,s);const error=[[0,log(norm(truth))]],residual=[[0,log(norm(r))]];
    for(let k=1;k<=steps;k++){const q=direction.map((v,i)=>v*sigma[i]),den=dot(q,q);if(den<1e-30)break;const alpha=gamma/den;x=x.map((v,i)=>v+alpha*direction[i]);r=r.map((v,i)=>v-alpha*q[i]);s=r.map((v,i)=>v*sigma[i]);const next=dot(s,s);error.push([k,log(norm(x.map((v,i)=>v-truth[i])))]);residual.push([k,log(norm(r))]);if(next<1e-30)break;direction=s.map((v,i)=>v+next/gamma*direction[i]);gamma=next;}
    const best=error.reduce((a,b)=>a[1]<b[1]?a:b);return {curves:[{name:'True state error',points:error},{name:'Training data residual',points:residual}],result:'Smallest displayed state error occurs at iteration '+best[0]+'. The truth is available only because this is a manufactured experiment; residual decrease alone cannot identify that stopping point.'};
  });
  add('deming-calibration',[['First-instrument noise SD',.1,1,.1,.5],['Second-instrument noise SD',.2,2,.2,1],['Horizontal units multiplier',.5,4,.5,1]],'Recorded first-channel value (display units)','Second-channel value',([sx,sy,units])=>{
    const x=[0,.9,2.2,2.7,4.3,4.8],y=[1.1,3.3,4.7,7.4,8.6,11.2],mx=x.reduce((a,b)=>a+b)/6,my=y.reduce((a,b)=>a+b)/6,X=x.map(v=>v-mx),Y=y.map(v=>v-my),sxx=dot(X,X),syy=dot(Y,Y),sxy=dot(X,Y),lambda=(sy/sx)**2,b=(syy-lambda*sxx+Math.sqrt((syy-lambda*sxx)**2+4*lambda*sxy*sxy))/(2*sxy),ordinary=sxy/sxx;
    return {curves:[{name:'Recorded pairs',points:x.map((v,i)=>[v*units,y[i]])},curve('Uncertainty-weighted Deming fit',xs(-.3,5.3),t=>my+b*(t-mx)),curve('Vertical least squares',xs(-.3,5.3),t=>my+ordinary*(t-mx))].map((c,i)=>i?{...c,points:c.points.map(([t,v])=>[t*units,v])}:c),result:'Physical Deming slope '+fmt(b)+'; displayed slope '+fmt(b/units)+'. The noise metric is transformed with the horizontal units, preserving the physical fit.'};
  });
  add('censored-lifetime',[['Test deadline (hours)',1,16,.5,7]],'Failure rate (1/hour)','Likelihood relative to its maximum',([deadline])=>{
    const durations=[2,4,6,9,14],observed=durations.map(t=>Math.min(t,deadline)),failed=durations.filter(t=>t<=deadline),d=failed.length,exposure=observed.reduce((a,b)=>a+b),failureTime=failed.reduce((a,b)=>a+b,0),rate=d/exposure,naive=d?d/failureTime:0,top=Math.max(3*rate,2*naive,5/exposure);
    const likelihood=(lambda,hours)=>d?lambda===0?0:Math.exp(d*Math.log(lambda/(d/hours))-hours*(lambda-d/hours)):Math.exp(-hours*lambda);
    const curves=[curve('All observed exposure',xs(0,top,180),x=>likelihood(x,exposure))];if(d)curves.push(curve('Discard unfinished tests',xs(0,top,180),x=>likelihood(x,failureTime)));
    return {curves,result:d?d+' failures in '+fmt(exposure)+' bearing-hours; rate estimate '+fmt(rate)+' per hour, mean lifetime '+fmt(1/rate)+' hours.': 'No failures. The one-sided 95% upper rate bound is '+fmt(-Math.log(.05)/exposure)+' per hour; the zero MLE does not prove infinite physical lifetime.'};
  });

  add('triangle-shape',[['Base length h',.1,1,.05,.5],['Altitude / base',.05,1,.05,.25]],'Horizontal coordinate x at y = ε/2','Field value',([h,ratio])=>{
    const altitude=h*ratio,gradient=[h,-h*h/(4*altitude)],exact=x=>x*x,linear=(x,y)=>gradient[0]*x+gradient[1]*y;
    const N=16,faces=[],lines=[];for(let i=0;i<N;i++)for(let j=0;j<N-i;j++){
      const point=(a,b)=>[h*(a/N+b/(2*N)),altitude*b/N];
      const a=point(i,j),b=point(i+1,j),c=point(i,j+1);
      faces.push({vertices:[a,b,c].map(([x,y])=>[x,y,exact(x)]),color:'#a6c8be'});
      if(i+j<N-1){const d=point(i+1,j+1);faces.push({vertices:[b,c,d].map(([x,y])=>[x,y,exact(x)]),color:'#a6c8be'});}
    }
    const vertices=[[0,0],[h,0],[h/2,altitude]];
    for(let j=0;j<=8;j++){const y=altitude*j/8,left=h*j/16,right=h-left;lines.push({points:xs(left,right,20).map(x=>[x,y,linear(x,y)]),color:'#ad491d',width:1.4});}
    lines.push({points:[...vertices,vertices[0]].map(([x,y])=>[x,y,exact(x)]),color:'#174d59',width:2});
    const l2=Math.sqrt(h*altitude/2*(h*h/6+h**4/(16*altitude**2)));
    return {curves:[curve('Exact u = x²',xs(h/4,3*h/4),exact),curve('Linear nodal interpolant at half altitude',xs(h/4,3*h/4),x=>linear(x,altitude/2))],result:'Interpolant gradient ('+fmt(gradient[0])+', '+fmt(gradient[1])+'). L² gradient-error norm '+fmt(l2)+'. A thin triangle creates a spurious transverse derivative even though u is independent of y.',scene:{faces,lines,points:vertices.map(([x,y],i)=>({position:[x,y,x*x],label:'Node '+(i+1)})),axisLabels:['x','y','field u'],caption:'Height represents the scalar field, not a third spatial coordinate. Green faces: exact quadratic field on a two-dimensional triangle. Orange wire lines: its affine nodal interpolant. Thin elements amplify the transverse gradient error.'}};
  });
  function bratu(a,n) {
    const exact=x=>2*Math.log(Math.cosh(a/2)/Math.cosh(a*(x-.5))),target=exact(.5),h=1/n;
    let u=xs(h,1-h,n-2).map(exact),lambda=2*a*a/Math.cosh(a/2)**2,residual=Infinity;
    for(let step=0;step<14;step++){
      const f=u.map((v,i)=>((u[i-1]||0)-2*v+(u[i+1]||0))/(h*h)+lambda*Math.exp(v));f.push(u[n/2-1]-target);residual=norm(f);if(residual<1e-10)break;
      const J=Array.from({length:n},()=>Array(n).fill(0));for(let i=0;i<n-1;i++){J[i][i]=-2/(h*h)+lambda*Math.exp(u[i]);if(i)J[i][i-1]=1/(h*h);if(i<n-2)J[i][i+1]=1/(h*h);J[i][n-1]=Math.exp(u[i]);}J[n-1][n/2-1]=1;
      const correction=solve(J,f.map(v=>-v));u=u.map((v,i)=>v+correction[i]);lambda+=correction[n-1];
    }return {u:[0,...u,0],lambda,residual,exact,reference:2*a*a/Math.cosh(a/2)**2};
  }
  add('pole-bracket',[['Crossing location',.05,.95,.01,.35],['Bisection rounds',5,30,1,20]],'Bisection update','log₁₀ width / residual magnitude',([location,n])=>{
    let lo=0,hi=1;const width=[],zero=[],pole=[];let hit=false;for(let j=0;j<n;j++){const mid=(lo+hi)/2,d=Math.abs(mid-location);width.push([j,log(hi-lo)]);zero.push([j,log(d)]);if(d<1e-14){hit=true;break;}pole.push([j,-Math.log10(d)]);if(mid<location)lo=mid;else hi=mid;}
    return {curves:[{name:'Bracket width',points:width},{name:'Continuous-zero residual',points:zero},...(pole.length?[{name:'Reciprocal-pole magnitude',points:pole}]:[])],result:'The sign-based brackets are identical for x − a and 1/(x − a), but only the continuous function has a root. '+(hit?'An exact midpoint hit makes the reciprocal undefined.':'The pole residual increases while the bracket contracts.')};
  });
  add('gaussian-ensemble',[['Selected carrier frequency ω',0,10,.1,2]],'Carrier frequency ω','Gaussian mean cosine response',([w])=>{
    const approx=t=>(2+Math.cos(Math.sqrt(1.5)*t))/3;
    return {curves:[curve('Continuous Gaussian integral',xs(0,10,240),t=>Math.exp(-t*t/4)),curve('Three-point Gauss–Hermite rule',xs(0,10,240),approx),{name:'Selected sampled response',points:[[w,approx(w)]]}],result:'At ω = '+fmt(w)+', three nodes give '+fmt(approx(w))+' versus exact '+fmt(Math.exp(-w*w/4))+'. Polynomial moment exactness through degree five does not control every oscillatory signal.'};
  });
  add('driven-decay',[['Decay rate (1/s)',.2,4,.2,.8],['Forcing frequency (Hz)',.25,3,.25,1],['Samples per forcing period',1,32,1,1]],'Time (s)','Temperature change (K)',([decay,f,n])=>{
    const w=2*Math.PI*f,h=1/(f*n),end=4/f,exact=t=>(decay*(Math.cos(w*t)-Math.exp(-decay*t))+w*Math.sin(w*t))/(decay*decay+w*w);let u=0;const points=[[0,0]];for(let j=0;j<4*n;j++){u=Math.exp(-decay*h)*u-Math.expm1(-decay*h)*Math.cos(w*j*h)/decay;points.push([(j+1)*h,u]);}
    return {curves:[curve('Exact convolution',xs(0,end,480),exact),{name:'Exact decay with frozen forcing',points}],result:'At '+n+' samples per cycle, endpoint error '+fmt(Math.abs(u-exact(end)))+' K. A single sample per period turns the oscillating drive into an apparently constant input.'};
  });
  function erf(x){const sign=Math.sign(x),a=Math.abs(x),t=1/(1+.3275911*a),p=(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t;return sign*(1-p*Math.exp(-a*a));}
  add('touching-solids',[['Left conductivity (W/m/K)',.5,8,.5,4],['Right conductivity (W/m/K)',.5,8,.5,1],['Elapsed time (s)',.1,8,.1,2]],'Position from contact (mm)','Temperature (°C)',([kl,kr,time])=>{
    const capacity=1e6,el=Math.sqrt(kl*capacity),er=Math.sqrt(kr*capacity),contact=(80*el+20*er)/(el+er),width=5*Math.sqrt(Math.max(kl,kr)/capacity*time)*1000;
    const temperature=(mm,t)=>{const x=mm/1000,k=x<=0?kl:kr,far=x<=0?80:20;return contact+(far-contact)*erf(Math.abs(x)/(2*Math.sqrt(k/capacity*t)));};
    return {curves:[curve('Selected elapsed time',xs(-width,width,240),x=>temperature(x,time)),curve('One quarter of that time',xs(-width,width,240),x=>temperature(x,time/4))],result:'Common interface temperature '+fmt(contact)+' °C; heat flux '+fmt(el*er/(el+er)*60/Math.sqrt(Math.PI*time))+' W/m². Both semi-infinite materials have volumetric heat capacity 10⁶ J/m³/K; contact resistance is zero.'};
  });
  add('adjoint-footprint',[['Sensor interpolation fraction',.05,.95,.05,.35],['Left smoothing weight',.05,.35,.05,.25]],'Parameter node on periodic 12-cell grid','Sensitivity to a unit sensor residual',([fraction,left])=>{
    const n=12,right=.4-left,b=Array(n).fill(0);b[6]=1-fraction;b[7]=fraction;const correct=b.map((v,i)=>.6*v+left*b[(i+n-1)%n]+right*b[(i+1)%n]),wrong=b.map((v,i)=>.6*v+left*b[(i+1)%n]+right*b[(i+n-1)%n]);
    return {curves:[curve('Exact transposed smoothing and sensor map',xs(0,n-1,n-1),(_,i)=>correct[i]),curve('Unreversed smoothing stencil',xs(0,n-1,n-1),(_,i)=>wrong[i])],result:'The sensitivity sums to '+fmt(correct.reduce((a,b)=>a+b,0))+'. Asymmetric smoothing must reverse its direction in the adjoint; the mismatch vanishes only at symmetric weights.'};
  });
  add('radiation-root',[['Implicit step duration (s)',.1,20,.1,1],['Initial temperature (K)',1.1,4,.1,2]],'Trial end temperature (K)','Backward-Euler residual (K)',([h,initial])=>{
    const residual=T=>T-initial+h*.0015*(T**4-1);let T=initial;for(let j=0;j<15;j++)T-=residual(T)/(1+h*.006*T**3);
    return {curves:[curve('Discrete cooling residual',xs(1,initial),residual),{name:'Computed implicit root',points:[[T,residual(T)]]}],result:'End temperature '+fmt(T)+' K, residual '+fmt(residual(T))+' K. Coefficient 0.0015 combines the same capacity, area and radiation constants as the linked OneSimulation lumped model.'};
  });
  add('heat-interpolation',[['Source intervals',1,32,1,4],['Polynomial profile power',1,3,1,2]],'Position along unit rod (m)','Temperature (K)',([n,p])=>{
    const f=x=>300+20*x**p,nodes=xs(0,1,n),values=nodes.map(f),interp=x=>{const j=Math.min(n-1,Math.floor(x*n)),t=x*n-j;return values[j]*(1-t)+values[j+1]*t;},integral=values.reduce((s,v,i)=>s+v*(i===0||i===n?.5:1),0)/n,exact=300+20/(p+1);
    return {curves:[curve('Source temperature law',xs(0,1,200),f),curve('Transferred piecewise-linear field',xs(0,1,2*n),interp)],result:'Integrated transferred field '+fmt(integral)+' K·m; exact integral '+fmt(exact)+' K·m; bias '+fmt(integral-exact)+' K·m. Refining the target samples alone does not change the source interpolant.'};
  });
  add('arrhenius-fit',[['Temperature span around 320 K',10,100,5,60],['Concentration noise SD',.002,.04,.002,.01]],'Temperature (K)','Concentration after 1 s',([span,sigma])=>{
    const temperatures=xs(320-span/2,320+span/2,9),trueEnergy=30000,R=8.314,trueP=[Math.log(1e5)-trueEnergy/(R*320),trueEnergy/(R*1000)],coordinate=T=>1000*(1/320-1/T),predict=(T,p)=>Math.exp(-Math.exp(p[0]+p[1]*coordinate(T))),data=temperatures.map((T,i)=>predict(T,trueP)+sigma*Math.sin((i+1)*2.2));let p=[trueP[0]+.1,trueP[1]*.9],H;
    for(let step=0;step<12;step++){const residual=temperatures.map((T,i)=>predict(T,p)-data[i]),J=temperatures.map(T=>{const X=coordinate(T),k=Math.exp(p[0]+p[1]*X),d=-k*Math.exp(-k);return[d,d*X];});H=[[0,0],[0,0]];const rhs=[0,0];for(let i=0;i<J.length;i++)for(let j=0;j<2;j++){rhs[j]-=J[i][j]*residual[i];for(let k=0;k<2;k++)H[j][k]+=J[i][j]*J[i][k];}const delta=solve(H,rhs);p=p.map((v,i)=>v+delta[i]);if(norm(delta)<1e-10)break;}
    const covariance=solve(H,[0,1]),se=sigma*Math.sqrt(covariance[1])*R;return {curves:[curve('True Arrhenius response',xs(320-span/2,320+span/2),T=>predict(T,trueP)),curve('Refitted response',xs(320-span/2,320+span/2),T=>predict(T,p)),{name:'Fixed-pattern noisy readings',points:temperatures.map((T,i)=>[T,data[i]])}],result:'Fitted activation energy '+fmt(p[1]*R)+' kJ/mol, local standard error '+fmt(se)+' kJ/mol; true energy 30 kJ/mol. Prefactor and energy are fitted jointly. The fixed sinusoidal perturbations isolate noise level and temperature span.'};
  });
  add('qr-tail',[['log₁₀ fitted component magnitude',0,12,1,8],['Residual-tail magnitude',.01,1,.01,.1]],'log₁₀ fitted component magnitude','Recovered residual norm',([selected,tail])=>({curves:[curve('Stored orthogonal tail',xs(0,12,120),()=>tail),curve('Difference of squared norms',xs(0,12,120),p=>Math.sqrt(Math.max(0,(10**p)**2+tail*tail-(10**p)**2)))],result:'At fitted magnitude 10^'+selected+', the subtractive residual is '+fmt(Math.sqrt(Math.max(0,(10**selected)**2+tail*tail-(10**selected)**2)))+' while the preserved tail is '+fmt(tail)+'. Binary64 rounding can erase the smaller squared term.'}));
  add('gmres-angle',[['Shear coupling',0,8,.2,2],['Initial residual angle (degrees)',0,180,5,90]],'Initial residual angle (degrees)','One-step residual / initial residual',([coupling,angle])=>{
    const reduction=t=>{const x=Math.cos(t*Math.PI/180),y=Math.sin(t*Math.PI/180),ax=x+coupling*y,ay=y,alpha=(x*ax+y*ay)/(ax*ax+ay*ay);return Math.hypot(x-alpha*ax,y-alpha*ay);};
    return {curves:[curve('One-dimensional GMRES correction',xs(0,180,180),reduction),{name:'Selected residual',points:[[angle,reduction(angle)]]}],result:'The shear matrix has both eigenvalues equal to one. Selected first-step residual factor '+fmt(reduction(angle))+'. At most two exact-arithmetic directions solve this nonsingular two-variable problem.'};
  });
  add('backward-error',[['Second equation scale',1,1000,1,500],['Second-component relative error',-.02,.02,.001,-.001]],'First-component relative error','Dimensionless componentwise backward error',([scale,e2])=>{
    const row=e=>Math.abs(e)/(Math.abs(1+e)+1),eta=e=>Math.max(row(e),row(e2));return {curves:[curve('Maximum componentwise backward error',xs(-.05,.05),eta)],result:'Second raw residual magnitude '+fmt(Math.abs(scale*e2))+', second row backward error '+fmt(row(e2))+'. The scale cancels from the row-normalised diagnostic although the raw residual changes.'};
  });
  add('coupled-newton',[['Initial u',.2,3,.1,1],['Initial v',0,4,.1,.8]],'Newton iteration','Balance residual norm',([u0,v0])=>{
    let u=u0,v=v0;const points=[];for(let k=0;k<=7;k++){const F=u+2*v-5,G=v-u*u;points.push([k,Math.hypot(F,G)]);const dv=(-G-2*u*F)/(1+4*u),du=-F-2*dv;u+=du;v+=dv;}
    return {curves:[{name:'Simultaneous two-balance Newton solve',points}],result:'Final state u = '+fmt(u)+', v = '+fmt(v)+'. The cross derivatives Fv = 2 and Gu = −2u enter the same linear correction; treating v as fixed changes the problem.'};
  });
  add('broyden-metric',[['Second step component',1,20,1,10],['Typical scale of second variable',1,20,1,10]],'Angle of a new trial step (degrees)','Predicted change from secant row',([s,scale])=>{
    const plain=[2/(1+s*s),2*s/(1+s*s)],weighted=[2/(1+s*s/scale**2),2*s/scale**2/(1+s*s/scale**2)],response=(b,t)=>b[0]*Math.cos(t*Math.PI/180)+b[1]*Math.sin(t*Math.PI/180);
    return {curves:[curve('Unscaled minimum-change update',xs(0,180),t=>response(plain,t)),curve('Scaled variable metric',xs(0,180),t=>response(weighted,t))],result:'Both rows predict change 2 for the observed step (1, '+s+'). They disagree in unobserved directions because the variable metric selects a different minimum-change extension.'};
  });
  add('complex-step',[['Evaluation point x',.5,4,.1,2]],'log₁₀ differentiation step h','log₁₀ derivative error',([x])=>({curves:[curve('Complex step on z³',xs(-14,-1),p=>log((3*x*x-(10**p)**2)-3*x*x)),curve('Centred real difference on x³',xs(-14,-1),p=>{const h=10**p;return log(((x+h)**3-(x-h)**3)/(2*h)-3*x*x);}),curve('Complex step applied to modulus |z|',xs(-14,-1),()=>0)],result:'Exact cubic derivative '+fmt(3*x*x)+'. For the nonanalytic modulus implementation the imaginary quotient is zero, so its error against d|x|/dx = 1 is exactly one at positive x.'}));
  add('gn-curvature',[['Residual curvature c',.1,1.5,.1,1],['Damping fraction α',.1,1,.05,.5]],'Iteration','Parameter x',([c,alpha])=>{
    let x=.1;const points=[[0,x]];for(let k=1;k<=18;k++){const gradient=x+2*c*x*(1+c*x*x),hessian=1+4*c*c*x*x;x-=alpha*gradient/hessian;points.push([k,x]);}
    return {curves:[{name:'Gauss–Newton for (x, 1 + cx²)',points}],result:'The optimum x = 0 is a strict minimum. The local damped factor is 1 − α(1 + 2c) = '+fmt(1-alpha*(1+2*c))+'. Its magnitude, not the existence of the minimum, predicts local attraction.'};
  });
  add('minimax-polynomial',[['Target monomial degree',2,8,1,3],['Interval half-width',.5,3,.5,1],['Leading monomial coefficient',.5,4,.5,1]],'Scaled interval coordinate x / h','Signed minimax residual',([n,h,coefficient])=>{
    const amplitude=coefficient*h**n/2**(n-1);return {curves:[curve('Scaled Chebyshev residual',xs(-1,1,240),x=>amplitude*Math.cos(n*Math.acos(x))),curve('Positive uniform bound',xs(-1,1),()=>amplitude),curve('Negative uniform bound',xs(-1,1),()=>-amplitude)],result:'For leading coefficient '+coefficient+' and degree '+n+' on an interval of half-width h, the best degree-'+(n-1)+' error is '+fmt(amplitude)+'. Lower-degree target terms and an interval translation do not change this bound. There are '+(n+1)+' alternating extreme points.'};
  });
  add('projection-gain',[['Middle reading',1,8,.2,3],['Last reading',1,10,.2,7]],'Reading index','Data / projected value',([middle,last])=>{
    const data=[2,middle,last],mean=(2+middle+last)/3,r=data.map(x=>x-mean),shapes=[[-1,0,1],[1,-2,1]],gain=shapes.map(v=>dot(r,v)**2/dot(v,v));return {curves:[{name:'Observed data',points:data.map((v,i)=>[i,v])},...shapes.map((v,j)=>({name:'Constant plus candidate '+(j+1),points:v.map((x,i)=>[i,mean+dot(r,v)/dot(v,v)*x])}))],result:'The two candidate squared-error reductions are '+gain.map(fmt).join(' and ')+'. Choose the larger gain only after using the same observation metric for both shapes.'};
  });
  add('endpoint-panel',[['Second-derivative bound M',1,20,1,10],['Selected endpoint panel width',.001,.05,.001,.005]],'Endpoint panel width h','Integral contribution / omitted-term bound',([M,chosen])=>({curves:[curve('Leading 2√h + 2h^(3/2)',xs(.001,.05),h=>2*Math.sqrt(h)+2*h**1.5),curve('Remainder bound M h^(5/2)/5',xs(.001,.05),h=>M*h**2.5/5)],result:'For g(x)/√x with g(0) = 1 and g′(0) = 3, the selected remainder bound is '+fmt(M*chosen**2.5/5)+'. The bound assumes |g″| ≤ M over that panel; the rest of the integral has a separate budget.'}));
  add('profile-target',[['Prior parameter correlation',-.9,.9,.1,.5],['Selected sum target z',2,10,.2,9]],'Target sum z = p₁ + p₂','Twice quadratic profile loss',([rho,z])=>{
    const variance=2+2*rho;return {curves:[curve('Both parameters refitted',xs(2,10),t=>(t-6)**2/variance),curve('Second parameter held at its mean',xs(2,10),t=>(t-6)**2/(1-rho*rho))],result:'Selected profiled state ('+fmt(2+(z-6)/2)+', '+fmt(4+(z-6)/2)+'). Sum variance '+fmt(variance)+'. Profiling moves nuisance parameters; fixing them gives a different curvature.'};
  });
  add('resolution-budget',[['Verified gradient bound (K/m)',.1,3,.1,2],['Independent noise SD (K)',.05,.5,.05,.1],['Cell spacing (m)',.1,1,.1,.5]],'Side weight w in (w, 1−2w, w)','Bias bound / 95% total half-width (K)',([L,sigma,spacing])=>{
    const bias=w=>2*Math.abs(w)*spacing*L,sd=w=>sigma*Math.sqrt(2*w*w+(1-2*w)**2);return {curves:[curve('Worst mixing bias',xs(-.2,.5),bias),curve('1.96 × output noise SD',xs(-.2,.5),w=>1.96*sd(w)),curve('Conservative combined half-width',xs(-.2,.5),w=>bias(w)+1.96*sd(w))],result:'The weights sum to one. Negative side weights sharpen some features but increase the absolute-distance bias bound and noise gain; a smaller variance alone does not establish a sharper reliable reconstruction.'};
  });
  add('constant-parasite',[['Second-start error δ',.001,.02,.001,.006],['Updates',4,30,2,12]],'Step number','State minus exact constant',([delta,n])=>({curves:[curve('BDF2 roots 1 and 1/3',xs(0,n,n),k=>1.5*delta*(1-3**(-k))),curve('Centred roots 1 and −1',xs(0,n,n),k=>k%2?delta:0)],result:'BDF2 damps its parasitic component but retains a limiting bias 3δ/2 = '+fmt(1.5*delta)+'. The centred method retains the even/odd oscillation.'}));
  add('discrete-adjoint',[['Decay parameter θ',.2,2,.1,1],['Euler steps to t = 0.5',2,40,2,2]],'Euler steps to fixed final time','Objective derivative',([theta,n])=>{
    const exactY=2*Math.exp(-theta*.5),exact=(exactY-1)*(-.5*exactY),gradient=N=>{const h=.5/N,factor=1-theta*h,y=2*factor**N;return (y-1)*(-2*N*h*factor**(N-1));};
    return {curves:[curve('Exact derivative of discrete objective',xs(2,40,19),gradient),curve('Continuous objective derivative',xs(2,40,19),()=>exact),{name:'Chosen grid',points:[[n,gradient(n)]]}],result:'At '+n+' Euler steps the derivative is '+fmt(gradient(n))+' versus continuum '+fmt(exact)+'. A correct discrete adjoint differentiates the numerical trajectory actually computed.'};
  });
  add('constraint-tension',[['Tangential speed (m/s)',0,4,.1,1],['Length (m)',.2,2,.1,.5],['Mass (kg)',.1,1,.1,.2]],'Angle from downward vertical (degrees)','Required inward tension (N)',([speed,length,mass])=>({curves:[curve('Bilateral radial constraint',xs(0,180,180),angle=>mass*(speed*speed/length+9.81*Math.cos(angle*Math.PI/180))),curve('Zero tension',xs(0,180),()=>0)],result:'At the bottom the required tension is '+fmt(mass*(speed*speed/length+9.81))+' N; at the top '+fmt(mass*(speed*speed/length-9.81))+' N. A string cannot supply negative tension: that part of the bilateral prediction corresponds to slack, not constrained motion.'}));
  add('anisotropic-face',[['Tensor cross term Kxy',-1.5,1.5,.1,1],['Neighbour offset dy (m)',-.1,.1,.01,0],['Longitudinal gradient gx (K/m)',-3,3,.2,2]],'Transverse gradient gy (K/m)','Flux through unit vertical face (W)',([cross,dy,gx])=>{
    const dx=.2,tau=4/dx,exact=gy=>-(4*gx+cross*gy),twopoint=gy=>-tau*(gx*dx+gy*dy);return {curves:[curve('Full tensor flux',xs(-4,4),exact),curve('Uncorrected two-point flux',xs(-4,4),twopoint)],result:'Correction coefficient Kxy − τdy = '+fmt(cross-tau*dy)+'. It vanishes when neighbour displacement aligns with K n, at dy = '+fmt(cross/20)+' m. Tensor entries Kxx = 4, Kyy = 2 keep all offered tensors positive definite.'};
  });
  add('checkerboard',[['Grid spacing h (m)',.02,.2,.02,.1]],'Fourier angle θ','Pressure Schur eigenvalue (1/m²)',([h])=>({curves:[curve('Collocated centred gradients',xs(0,Math.PI,200),t=>Math.sin(t)**2/(h*h)),curve('Staggered face gradients',xs(0,Math.PI,200),t=>4*Math.sin(t/2)**2/(h*h))],result:'At θ = π the centred symbol vanishes while the staggered symbol is '+fmt(4/(h*h))+' m⁻². Removing the pressure mean does not remove this alternating zero-mean mode.'}));
  add('pivoted-load',[['Leading entry a',-.5,.5,.05,0],['First load b₁',0,6,.2,4],['Second load b₂',0,6,.2,5]],'Leading matrix entry a','Solved state component',([a,b1,b2])=>{
    const state=t=>solve([[t,2],[3,1]],[b1,b2]),chosen=state(a);return {curves:[curve('First component',xs(-.5,.5),t=>state(t)[0]),curve('Second component',xs(-.5,.5),t=>state(t)[1])],result:'For A = ((a,2),(3,1)), selected solution ('+chosen.map(fmt).join(', ')+'). Determinant a − 6 = '+fmt(a-6)+' stays nonzero even at a = 0; a row exchange avoids the zero leading pivot.'};
  });
  add('chain-factors',[['Additional load on last node',0,6,.2,1]],'Additional last-node load','Displacement increment',([load])=>{
    const A=[[4,-1,0],[-1,3,-1],[0,-1,2]],baseline=solve(A,[2,4,0]),response=solve(A,[0,0,1]);return {curves:response.map((v,i)=>curve('Node '+(i+1),xs(0,6),f=>f*v)),result:'Pivots 4, 11/4 and 18/11 can be reused. At the selected load the solution is ('+baseline.map((v,i)=>fmt(v+load*response[i])).join(', ')+'); the unit response is (1/18, 2/9, 11/18).'};
  });
  add('binary-sum',[['Significant binary bits',3,16,1,4],['Repeated small addend',.125,1,.125,.5]],'Small terms accumulated after 8','Accumulated sum',([bits,small])=>{
    function round(x){if(!x)return 0;const step=2**(Math.floor(Math.log2(Math.abs(x)))+1-bits),q=x/step,lo=Math.floor(q),fraction=q-lo;return (fraction>.5?lo+1:fraction<.5?lo:lo%2?lo+1:lo)*step;}
    let total=8;const points=[[0,total]];for(let n=1;n<=20;n++){total=round(total+small);points.push([n,total]);}
    return {curves:[{name:'Round after every addition',points},curve('Accumulate small terms exactly first',xs(0,20,20),n=>round(8+n*small)),curve('Exact real sum',xs(0,20,20),n=>8+n*small)],result:'Round-to-nearest, ties-to-even with '+bits+' significant binary bits; spacing above 8 is '+fmt(2**(4-bits))+'. Only intermediate arithmetic changes between the first two curves.'};
  });
  add('weighted-constant',[['Third reading',2,12,.2,8],['Third-reading SD / other SD',.5,5,.5,2]],'Relative standard deviation of third reading','Fitted constant',([third,ratio])=>{
    const fit=s=>(3+third/(s*s))/(2+1/(s*s));return {curves:[curve('Inverse-variance weighted fit',xs(.5,5),fit),curve('Equal-weight fit',xs(.5,5),()=>(3+third)/3)],result:'Readings are (1,2,'+fmt(third)+'). At the selected uncertainty ratio the fit is '+fmt(fit(ratio))+'. The third weight scales as inverse variance, not inverse standard deviation.'};
  });
  add('centred-design',[['Coordinate offset',0,2000,50,1000]],'Coordinate offset','log₁₀ two-norm condition of design',([offset])=>{
    function condition(c){const trace=5+3*c*c,det=6,large=(trace+Math.sqrt(trace*trace-4*det))/2;return Math.sqrt(large*large/det);}
    return {curves:[curve('Columns 1 and shifted position',xs(0,2000),c=>Math.log10(condition(c))),curve('Columns 1 and centred position',xs(0,2000),()=>Math.log10(Math.sqrt(3/2)))],result:'At offset '+offset+', design condition '+fmt(condition(offset))+'. Centring positions (c−1,c,c+1) makes the Gram matrix diag(3,2), preserving the set of fitted straight lines.'};
  });
  add('householder',[['Direction of reflected input (degrees)',0,180,5,90]],'Original direction (degrees)','Reflected vector component',([angle])=>{
    const H=[[-.6,-.8],[-.8,.6]],component=(t,j)=>H[j][0]*Math.cos(t*Math.PI/180)+H[j][1]*Math.sin(t*Math.PI/180);return {curves:[curve('Reflected first component',xs(0,180),t=>component(t,0)),curve('Reflected second component',xs(0,180),t=>component(t,1))],result:'The reflector sends (3,4) to (−5,0). At the selected unit-vector angle its output is ('+fmt(component(angle,0))+', '+fmt(component(angle,1))+') with norm one; all angles between pairs are preserved.'};
  });
  add('nullspace-sum',[['Identified sum a+b',1,6,.2,3],['Quadratic weight on second parameter',.2,5,.2,1]],'Free first coordinate a','Second coordinate / squared parameter norm',([sum,weight])=>{
    const optimum=weight*sum/(1+weight);return {curves:[curve('Exact-solution family b = sum − a',xs(-1,7),a=>sum-a),curve('Weighted squared norm / sum²',xs(-1,7),a=>(a*a+weight*(sum-a)**2)/(sum*sum))],result:'The minimum weighted norm occurs at ('+fmt(optimum)+', '+fmt(sum-optimum)+'). The data identify the sum; the choice within its nullspace comes from the stated parameter metric. The norm curve is normalised by sum².'};
  });
  add('scalar-refinement',[['Approximate factor relative error',-.8,.8,.05,.25],['Initial state error',-.5,.5,.05,.2]],'Refinement step','Relative state error',([error,start])=>{
    const factor=error/(1+error);return {curves:[curve('Residual correction with approximate factor',xs(0,12,12),k=>start*factor**k)],result:'Using approximate scalar factor â = a(1 + ε), each correction multiplies the error by ε/(1 + ε) = '+fmt(factor)+'. Accurate residual formation permits improvement only when this factor has magnitude below one.'};
  });
  add('mapped-gauss',[['Interval left endpoint',0,2,.2,1],['Interval length',.5,5,.5,3],['Monomial degree',0,6,1,2]],'Gauss nodes','Integral / exact reference',([left,length,power])=>{
    const exact=((left+length)**(power+1)-left**(power+1))/(power+1),counts=[1,2,3,4,5];return {curves:[curve('Mapped Gauss rule',counts,n=>{const q=gauss(n);return length/2*q.nodes.reduce((s,v,i)=>s+q.weights[i]*(left+length*(v+1)/2)**power,0);}),curve('Exact antiderivative',counts,()=>exact)],result:'Physical quadrature weights include interval half-length '+fmt(length/2)+'. For this degree-'+power+' monomial, n nodes are exact once 2n−1 ≥ '+power+'.'};
  });
  add('element-underintegration',[['Right nodal value',-2,2,.1,-1],['Element length',.2,3,.2,1]],'Reference position x/h','Interpolated field / squared field',([right,length])=>{
    const exact=length*(1+right+right*right)/3,midpoint=length*((1+right)/2)**2;return {curves:[curve('Linear field',xs(0,1),t=>1+(right-1)*t),curve('Squared field',xs(0,1),t=>(1+(right-1)*t)**2)],result:'Exact squared norm '+fmt(exact)+', midpoint quadrature '+fmt(midpoint)+'. Two-point Gauss recovers the exact quadratic integral. An antisymmetric nonzero field can be invisible at the midpoint.'};
  });
  add('horner-remainder',[['Trial divisor root a',.5,3.5,.1,2.1]],'Evaluation point x','Polynomial / deflated reconstruction',([a])=>{
    const q1=a-6,q0=11+a*q1,remainder=-6+a*q0,p=x=>((x-6)*x+11)*x-6;return {curves:[curve('Original polynomial',xs(0,4),p),curve('Discarded-remainder reconstruction',xs(0,4),x=>(x-a)*(x*x+q1*x+q0))],result:'Quotient x² + ('+fmt(q1)+')x + '+fmt(q0)+', remainder '+fmt(remainder)+'. The derivative at a is '+fmt(3*a*a-12*a+11)+'. Dropping the remainder shifts the entire polynomial by a constant.'};
  });
  add('cg-cube',[['Cube mesh intervals',4,12,2,6],['log₁₀ relative CG tolerance',-10,-1,1,-5]],'CG iteration','log₁₀ relative residual / relative field error',([N,power])=>{
    const m=N-1,size=m*m*m,scale=N*N;
    function apply(v){return v.map((u,id)=>{const i=id%m,j=Math.floor(id/m)%m,k=Math.floor(id/(m*m));return scale*(6*u-(i?v[id-1]:0)-(i<m-1?v[id+1]:0)-(j?v[id-m]:0)-(j<m-1?v[id+m]:0)-(k?v[id-m*m]:0)-(k<m-1?v[id+m*m]:0));});}
    function run(tol,record,reference){let u=Array(size).fill(0),r=Array(size).fill(1),p=r.slice(),rr=size;const residual=[[0,0]],error=[[0,0]];for(let k=1;k<=150;k++){const ap=apply(p),den=dot(p,ap);if(den<1e-28)break;const alpha=rr/den;u=u.map((v,i)=>v+alpha*p[i]);r=r.map((v,i)=>v-alpha*ap[i]);const next=dot(r,r),relative=Math.sqrt(next/size);residual.push([k,log(relative)]);if(record)error.push([k,log(norm(u.map((v,i)=>v-reference[i]))/norm(reference))]);if(relative<tol)break;p=r.map((v,i)=>v+next/rr*p[i]);rr=next;}return {u,residual,error};}
    const ref=run(1e-12,false),chosen=run(10**power,true,ref.u),difference=Math.max(...chosen.u.map((v,i)=>Math.abs(v-ref.u[i])));
    return {curves:[{name:'Relative algebraic residual',points:chosen.residual},{name:'Relative nodal field error',points:chosen.error}],result:'For −Δu = 1 in a grounded unit cube: reference peak '+fmt(Math.max(...ref.u))+'; selected CG peak '+fmt(Math.max(...chosen.u))+'; maximum difference '+fmt(difference)+'. This browser companion uses a seven-point finite-difference grid; the linked OneSimulation laboratory uses trilinear finite elements, so compare convergence behaviour rather than identical nodal values.'};
  });
  add('reaction-order',[['Reaction rate k (1/s)',.2,2,.1,.8],['End time (s)',1,8,.5,6],['Maximum steps',20,160,20,80]],'Integration steps','log₁₀ maximum concentration error',([k,end,n])=>{
    const counts=xs(10,n,(n-10)/10);function error(N,rk){const h=end/N,z=-k*h,factor=rk?1+z+z*z/2+z**3/6+z**4/24:1+z;let state=1,worst=0;for(let i=1;i<=N;i++){state*=factor;worst=Math.max(worst,Math.abs(state-Math.exp(-k*i*h)));}return log(worst);}
    return {curves:[curve('Forward Euler',counts,N=>error(N,false)),curve('Classical RK4',counts,N=>error(N,true))],result:'The model transfers A to B with A + B = 1 and initial A = 1. Both methods conserve that sum; time accuracy differs. RK4 costs four slope evaluations per step.'};
  });
  add('wave-dispersion',[['Courant number ν',.1,1,.05,.5]],'Fourier angle kΔx','Numerical / physical wave frequency',([nu])=>({curves:[curve('Centred space, leapfrog time',xs(.01,Math.PI,200),t=>2*Math.asin(nu*Math.sin(t/2))/(nu*t)),curve('Centred space, exact time',xs(.01,Math.PI,200),t=>2*Math.sin(t/2)/t),curve('Continuum reference',xs(.01,Math.PI,200),()=>1)],result:'The formula follows sin(ωh/2) = ν sin(kΔx/2). At ν = 1, this one-dimensional constant-speed scheme has exact resolved-mode phase propagation.'}));
  add('manufactured-poisson',[['Mesh intervals per axis',8,40,4,16],['Sine half-waves per direction',1,3,1,1]],'Horizontal position x','Scalar field on selected grid row',([N,mode])=>{
    const m=N-1,h=1/N,size=m*m,k=mode*Math.PI,exact=(x,y)=>Math.sin(k*x)*Math.sin(k*y),diag=Array(size),b=Array(size);
    for(let j=0;j<m;j++)for(let i=0;i<m;i++){const x=(i+1)*h,y=(j+1)*h,id=j*m+i,a=1+x;diag[id]=4*a/(h*h)+2;b[id]=(2*a*k*k+2)*exact(x,y)-k*Math.cos(k*x)*Math.sin(k*y);}
    function apply(u){return u.map((v,id)=>{const i=id%m,j=Math.floor(id/m),a=1+(i+1)*h;return diag[id]*v-((i?(a-h/2)*u[id-1]:0)+(i<m-1?(a+h/2)*u[id+1]:0)+(j?a*u[id-m]:0)+(j<m-1?a*u[id+m]:0))/(h*h);});}
    let u=Array(size).fill(0),r=b.slice(),z=r.map((v,i)=>v/diag[i]),p=z.slice(),rz=dot(r,z),iterations=0;
    for(;iterations<400;iterations++){const ap=apply(p),den=dot(p,ap);if(den<1e-28)break;const alpha=rz/den;u=u.map((v,i)=>v+alpha*p[i]);r=r.map((v,i)=>v-alpha*ap[i]);if(norm(r)<1e-10*Math.max(1,norm(b)))break;z=r.map((v,i)=>v/diag[i]);const next=dot(r,z);p=z.map((v,i)=>v+next/rz*p[i]);rz=next;}
    const row=Math.max(1,Math.round(N/(2*mode))),y=row*h,values=[0,...u.slice((row-1)*m,row*m),0],maxError=Math.max(...u.map((v,id)=>Math.abs(v-exact((id%m+1)*h,(Math.floor(id/m)+1)*h))));
    return {curves:[curve('Variable-coefficient five-point solve',xs(0,1,N),(_,i)=>values[i]),curve('Manufactured continuum field',xs(0,1,240),x=>exact(x,y))],result:'For −∇·[(1+x)∇u] + 2u = f, maximum nodal error '+fmt(maxError)+', relative algebraic residual '+fmt(norm(r)/norm(b))+'. Slice y = '+fmt(y)+'; '+(iterations+1)+' Jacobi-preconditioned CG iterations. Refinement measures discretisation after the algebraic solve has converged.'};
  });
  add('bratu-branch',[['Branch parameter a',.3,5,.1,2.4],['Mesh intervals',8,40,4,16]],'Position x','Temperature u',([a,n])=>{
    const solution=bratu(a,n),rows=xs(.3,5,24),positions=xs(0,1,24),faces=[];
    const point=(x,b)=>[x,2*b*b/Math.cosh(b/2)**2,2*Math.log(Math.cosh(b/2)/Math.cosh(b*(x-.5)))];
    for(let i=0;i<positions.length-1;i++)for(let j=0;j<rows.length-1;j++)faces.push({vertices:[point(positions[i],rows[j]),point(positions[i+1],rows[j]),point(positions[i+1],rows[j+1]),point(positions[i],rows[j+1])],color:j<11?'#9dc6b9':'#d9b291'});
    return {curves:[curve('Analytical fixed-midpoint branch',xs(0,1),solution.exact),curve('Augmented finite-difference solve',xs(0,1,n),(_,i)=>solution.u[i])],result:'Exact load λ = '+fmt(solution.reference)+', discrete λ = '+fmt(solution.lambda)+', augmented residual '+fmt(solution.residual)+'. Matching midpoint temperature allows comparison on both sides of the fold.',scene:{faces,lines:[{points:solution.u.map((u,i)=>[i/n,solution.lambda,u]),color:'#ad491d',width:3}],points:[],axisLabels:['x','load λ','temperature u'],caption:'A three-dimensional graph of a family of one-dimensional boundary-value solutions. The folded sheet shows two temperature profiles at some loads; its vertical coordinate is temperature. The highlighted line is the computed finite-difference branch at the selected midpoint temperature.'}};
  });
  // Sweep controls that select an inspection point should also move a visible mark.
  const inspection = (key, extra) => {
    const model=studies['complete-numerics-'+key],calculate=model.calculate;
    model.calculate=values=>{const data=calculate(values);data.curves.push(...extra(values));return data;};
  };
  const mark=(name,x,y)=>({name,points:[[x,y]]});
  inspection('schur-star',([a,b,c])=>[mark('Selected third conductance',c,a*b/(a+b+c))]);
  inspection('edge-leverage',([g,bridge])=>[mark('Selected AB resistance / 1 Ω',g,1/(g+.5)),
    mark('Selected AB leverage',g,g/(g+.5)),curve('Bridge resistance / 1 Ω',xs(.05,10),()=>1/bridge)]);
  inspection('qr-tail',([power,tail])=>[mark('Selected subtractive residual',power,Math.sqrt(Math.max(0,(10**power)**2+tail*tail-(10**power)**2)))]);
  inspection('stiff-amplification',([end,r])=>[mark('Selected exact amplification',r,Math.exp(-r)),
    mark('Selected backward Euler',r,1/(1+r)),mark('Selected trapezoidal',r,(1-r/2)/(1+r/2))]);
  inspection('profile-target',([rho,z])=>[mark('Selected profiled target',z,(z-6)**2/(2+2*rho))]);
  inspection('endpoint-panel',([M,h])=>[mark('Selected remainder bound',h,M*h**2.5/5)]);
  inspection('decay-design',([k,sigma,t])=>[mark('Selected observation time',t,(t*Math.exp(-k*t)/sigma)**2)]);
  inspection('contact-resistance',([a,b,c])=>[mark('Selected contact fraction',c,c/(a+b+c))]);
  inspection('slope-window',([sigma,cubic,h])=>[mark('Selected three-point error',h,Math.hypot(cubic*h*h,sigma/(h*Math.sqrt(2)))),
    mark('Selected five-point error',h,Math.hypot(3.4*cubic*h*h,sigma/(h*Math.sqrt(10))))]);
})();
