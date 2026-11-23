"""Plots of the derived coordinate, material, rotating-source and relativistic laws."""
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.4, 4), layout="constrained")
angle = np.linspace(0, 2*np.pi, 401)
for radius in (1, 2, 3):
    ax.plot(radius*np.cos(angle), radius*np.sin(angle), color="#b2c8ca")
for phi in np.arange(0, 2*np.pi, np.pi/6):
    ax.plot([0, 3*np.cos(phi)], [0, 3*np.sin(phi)], color="#b2c8ca")
phi = np.pi/4
base = np.array([2*np.cos(phi), 2*np.sin(phi)])
for direction, label, color in [(base/2, "radial", "#0b6574"), (np.array([-base[1], base[0]])/2, "azimuthal", "#bd653e")]:
    ax.quiver(*base, *direction, angles="xy", scale_units="xy", scale=1, color=color)
    ax.text(*(base+1.15*direction), label, color=color)
ax.set(aspect="equal", xlabel="x", ylabel="y", xlim=(-3.2, 3.5), ylim=(-3.2, 3.5))
fig.savefig("electrodynamics-coordinates.png", dpi=170)
plt.close(fig)

temperature = np.linspace(100, 1500, 301)
fig, ax = plt.subplots(figsize=(6.4, 4), layout="constrained")
ax.plot(temperature, np.full_like(temperature, -2e-5), label="Orbital contribution")
ax.plot(temperature, .012/temperature, label="Orientational contribution")
ax.plot(temperature, -2e-5+.012/temperature, label="Sum")
ax.axhline(0, color="0.4", linewidth=.6)
ax.set(xlabel="Temperature (K)", ylabel="Susceptibility")
ax.legend(); ax.grid(alpha=.2)
fig.savefig("electrodynamics-magnetic-response.png", dpi=170)
plt.close(fig)

theta=np.linspace(0, np.pi, 301)
p2=(3*np.cos(theta)**2-1)/2
fig, ax=plt.subplots(figsize=(6.4, 4),layout="constrained")
ax.plot(np.degrees(theta),-p2/3,label="Conducting magnet")
ax.plot(np.degrees(theta),-p2/5,label="Insulating magnet")
ax.set(xlabel="Polar angle (degrees)",ylabel="Surface potential / (Ω B₀ a²)")
ax.legend();ax.grid(alpha=.2)
fig.savefig("electrodynamics-rotating-magnet.png",dpi=170)
plt.close(fig)

z=np.linspace(0,3,301)
fig, axes=plt.subplots(1,2,figsize=(9,3.8),layout="constrained")
axes[0].plot(z,np.cos(z)-1,label="Exact delay, real")
axes[0].plot(z,-z*z/2,"--",label="Taylor, real")
axes[1].plot(z,np.sin(z),label="Exact delay, imaginary")
axes[1].plot(z,z,"--",label="Taylor, imaginary")
for ax in axes:
    ax.set(xlabel="Frequency × shell light-crossing time",ylabel="Memory factor")
    ax.legend();ax.grid(alpha=.2)
fig.savefig("electrodynamics-self-force.png",dpi=170)
plt.close(fig)

fig, ax=plt.subplots(figsize=(6.4,4),layout="constrained")
ct=np.linspace(0,5,101)
ax.plot(ct,ct,"--",color="gray",label="Light")
ax.plot(-ct,ct,"--",color="gray")
ax.plot(.6*ct,ct,label="Moving clock, β = 0.6")
x=np.linspace(-3,3,101)
ax.plot(x,.6*x+2,label="One moving-frame simultaneity line")
ax.axvline(0,color="0.3",linewidth=.7)
ax.set(xlabel="x (light-seconds)",ylabel="ct (light-seconds)",xlim=(-3,3),ylim=(0,5),aspect="equal")
ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.savefig("electrodynamics-spacetime.png",dpi=170)
plt.close(fig)
