"""Grounded sphere: a 3D source-collocation solve and an independent image solution."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from onesim.numerics.linear.matrices.dense_linear_algebra import solve_dense

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sources', type=int, default=48)
parser.add_argument('--distance', type=float, default=2.0, help='Real-charge distance / sphere radius')
parser.add_argument('--source-radius', type=float, default=.65, help='Fictitious-source radius / sphere radius')
args = parser.parse_args()
if not 24 <= args.sources <= 96 or not 1.5 <= args.distance <= 3 or not .45 <= args.source_radius <= .85:
    parser.error('Use 24–96 sources, distance 1.5–3, and source radius 0.45–0.85')

def fibonacci_sphere(n, offset=0.0):
    j = np.arange(n)
    z = 1 - 2*(j+.5)/n
    phi = j*np.pi*(3-np.sqrt(5))+offset
    return np.column_stack([np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z])

boundary = fibonacci_sphere(args.sources)
sources = args.source_radius*boundary
charge = np.array([0.,0.,args.distance])
influence = 1/np.linalg.norm(boundary[:,None,:]-sources[None,:,:],axis=2)
rhs = -1/np.linalg.norm(boundary-charge,axis=1)
weights = np.asarray(solve_dense(influence,rhs))
factor = 8.9875517923  # q/(4*pi*eps0), for q=1 nC and radius=1 m

def potential(points):
    points=np.atleast_2d(points)
    return factor*(1/np.linalg.norm(points-charge,axis=1)
                   + (1/np.linalg.norm(points[:,None,:]-sources[None,:,:],axis=2))@weights)

def exact(points):
    points=np.atleast_2d(points)
    image=np.array([0.,0.,1/args.distance])
    return factor*(1/np.linalg.norm(points-charge,axis=1)
                   -1/args.distance/np.linalg.norm(points-image,axis=1))

def field(point):
    real=point-charge
    delta=point-sources
    return real/np.linalg.norm(real)**3 + (weights[:,None]*delta/np.linalg.norm(delta,axis=1)[:,None]**3).sum(axis=0)

check = fibonacci_sphere(1201,.37)
boundary_error=float(np.max(np.abs(potential(check))))
algebraic_error=float(np.max(np.abs(influence@weights-rhs)))
print(f'Sources: {args.sources}; d/a={args.distance:g}; source radius/a={args.source_radius:g}')
print(f'Linear residual: {algebraic_error:.3e}; withheld boundary error: {boundary_error:.6e} V')
print(f'Induced charge: {weights.sum():.8f} nC; exact: {-1/args.distance:.8f} nC')
assert algebraic_error < 1e-9

fig = plt.figure(figsize=(11,4.8),layout='constrained')
ax=fig.add_subplot(121,projection='3d')
theta=np.linspace(0,np.pi,25);phi=np.linspace(0,2*np.pi,49)
T,P=np.meshgrid(theta,phi)
ax.plot_surface(np.sin(T)*np.cos(P),np.sin(T)*np.sin(P),np.cos(T),color='#94b8af',alpha=.85,linewidth=0)
for angle in [np.pi/6,np.pi/2,5*np.pi/6,np.pi]:
    for azimuth in [0,np.pi/2,np.pi,3*np.pi/2]:
        p=charge+.08*np.array([np.sin(angle)*np.cos(azimuth),np.sin(angle)*np.sin(azimuth),np.cos(angle)])
        path=[p.copy()]
        for _ in range(160):
            e=field(p);p=p+.035*e/np.linalg.norm(e);path.append(p.copy())
            if np.linalg.norm(p)<1.015 or np.linalg.norm(p)>4.4:break
        curve=np.array(path);ax.plot(*curve.T,color='#ad491d',linewidth=.8)
ax.scatter(*charge,color='#ad491d',s=30)
ax.set(xlabel='x/a',ylabel='y/a',zlabel='z/a',title='Computed exterior field lines')
ax.set_box_aspect([1,1,1]);ax.set(xlim=(-2.5,2.5),ylim=(-2.5,2.5),zlim=(-2,4))
ax=fig.add_subplot(122)
angles=np.linspace(0,np.pi,181)
for azimuth in [0,np.pi/2]:
    points=np.column_stack([np.sin(angles)*np.cos(azimuth),np.sin(angles)*np.sin(azimuth),np.cos(angles)])
    ax.plot(np.rad2deg(angles),potential(points),label=f'Collocation, φ={np.rad2deg(azimuth):g}°')
ax.axhline(0,color='black',linestyle='--',label='Exact image solution')
ax.set(xlabel='Polar angle θ (degrees)',ylabel='Boundary potential error (V)',title='Check locations outside the fit')
ax.legend();ax.grid(alpha=.2)
fig.savefig('complete-spatial-sphere.png',dpi=170)
