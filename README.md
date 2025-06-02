# SPARSE: Scattering Poles and Amplitudes from Radial Schrödinger Equations

This algorithm calculates scattering amplitudes from a two-body, multi-channel, radial Schrödinger equation in the form:

$$\left(-\frac{1}{2 \mu} \frac{\mathrm{d}^2}{\mathrm{d}r^2} + \frac{L (L + 1)}{2 \mu r^2} + V(r) \right) \psi(r) = E \psi(r)$$

where $r$ is the relative distance between the two degrees of freedom,
$\psi(r)=(\psi_1(r),\psi_2(r),\dots,\psi_N(r))$ is a wave function with N channels,
$\mu = \mathrm{diag}(\mu_1, \mu_2, \dots, \mu_N)$ is a diagonal reduced-mass matrix,
$L=\mathrm{diag}(L_1, L_2, \dots, L_N)$ is a diagonal orbital-angular-momentum matrix,
and $V(r)$ is a $N \times N$ potential matrix with both diagonal and off-diagonal entries that depends on $r$.

We define the threshold matrix as the limit of the potential matrix for $r\to\infty$: $T=\lim_{r\to\infty}V(r)$.
This module focuses on cases where all limits are well defined,
the threshold matrix is diagonal, $T=\mathrm{diag}(T_1,T_2,\dots,T_N)$, and each threshold is either a finite number or $+\infty$.
The Schrödinger equation then admits scattering solutions for energies larger than the lowest threshold, $E \geq \min T$.

The user...
