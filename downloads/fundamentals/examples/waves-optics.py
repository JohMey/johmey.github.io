"""Propagate two Gaussian fields with the same measured input intensity."""
import matplotlib.pyplot as plt
import numpy as np
from onesim.physics.optics.propagation.numerics.angular_spectrum import (
    AngularSpectrumPropagation, longitudinal_wavenumber,
)

wavelength, width = 633e-9, 0.40e-3  # m; width is the 1/e field radius
k = 2 * np.pi / wavelength
rayleigh = np.pi * width**2 / wavelength
x = np.linspace(-6e-3, 6e-3, 2048, endpoint=False)
kx = 2 * np.pi * np.fft.fftfreq(len(x), d=x[1] - x[0])
propagate = AngularSpectrumPropagation(longitudinal=longitudinal_wavenumber(kx[:, None], k),
    forward=np.fft.fft, inverse=np.fft.ifft, reference_wavenumber=k)
z = np.linspace(0, 0.90, 91)
fig, ax = plt.subplots(figsize=(6.6, 3.5), layout="constrained")
for curvature, label in [(-0.60, "Converging"), (0.60, "Diverging")]:
    field = np.exp(-(x / width)**2 + 1j * k * x**2 / (2 * curvature))
    radii = []
    for distance in z:
        intensity = np.abs(propagate(field, distance))**2
        radii.append(2 * np.sqrt(np.average(x**2, weights=intensity)))
    reference = width * np.sqrt((1 + z / curvature)**2 + (z / rayleigh)**2)
    assert np.allclose(radii, reference, rtol=2e-5, atol=0)
    print(f"{label}: narrowest sampled radius {1000*min(radii):.3f} mm at {z[np.argmin(radii)]:.2f} m")
    ax.plot(z, 1000 * np.asarray(radii), label=label)
ax.set(xlabel="Propagation distance (m)", ylabel="Beam radius (mm)")
ax.grid(alpha=0.2)
ax.legend()
fig.savefig("waves-optics.png", dpi=180)
