# SPARSE: Scattering Poles and Amplitudes from Radial Schrödinger Equations

**If this program has been significant in your research, and you wish to acknowledge the project in your publication, I suggest citing the following preprint:**  

*R. Bruschini, SPARSE: Scattering Poles and Amplitudes from Radial Schrödinger Equations, arXiv:2507.01002 [quant-ph].*  

This algorithm calculates scattering amplitudes from a two-body, multi-channel, radial Schrödinger equation in the form:

$$\left(-\frac{1}{2 \mu} \frac{\mathrm{d}^2}{\mathrm{d}r^2} + \frac{L (L + 1)}{2 \mu r^2} + V(r) - E \right) u(r) = 0$$

where $r$ is the relative distance between the two degrees of freedom,
$u(r)=(u_1(r),u_2(r),\dots,u_N(r))$ is a reduced radial wave function with N channels,
$\mu = \mathrm{diag}(\mu_1, \mu_2, \dots, \mu_N)$ is a diagonal reduced-mass matrix,
$L=\mathrm{diag}(L_1, L_2, \dots, L_N)$ is a diagonal orbital-angular-momentum matrix,
and $V(r)$ is a $N \times N$ potential matrix with both diagonal and off-diagonal elements that depends on $r$.
Note that we used natural units in which $\hbar=c=1$.

We define the threshold matrix as the limit of the potential matrix for $r\to\infty$: $T=\lim_{r\to\infty}V(r)$.
This module focuses on cases where all limits are well defined,
the threshold matrix is diagonal, $T=\mathrm{diag}(T_1,T_2,\dots,T_N)$, and each threshold is either a finite number or $+\infty$.
The Schrödinger equation then admits scattering solutions for energies larger than the lowest threshold, $E \geq \min T$.

SPARSE solves the Schrödinger equation numerically using the finite difference method by sampling the continuous distance $r$ on a
grid of equally-spaced points. The grid begins at $r_\text{min}=0$, contains $M$ nodes separated by a constant distance $d$ in its interior
$r_n=n d$ with $n=1,\dots,M$, and ends at $r_\text{max}=(M+1) d$.
The reduced radial wave function $u(r)$ with N channels is treated as a numerical array containing its values at each of the M nodes in the interior of $r$.
It is an array with $N \times M$ entries. Similarly, the potential matrix $V$ is treated as an array with $N \times N \times M$ entries, corresponding to
each potential matrix element evaluated at each node in the interior of $r$.

## Setup

The numerical Schrödinger equation is completely specified by 4 quantities:  
1. the channels' orbital angular momenta;
2. the channels' thresholds;
3. the channels' reduced masses;
4. the numerical potential matrix.

The channels' specifics (orbital angular momenta, thresholds, reduced masses) must be stored in a CSV file with 1 line for the header followed by $N$ lines for the values.
Each line must contain at least 3 entries.
The header must contain at least the following strings: "l", "mu", and "threshold".
The following $N$ lines must contain at least the corresponding values for the channels $i=1,\dots,N$.
An entry under "l" is an integer indicating the orbital angular momentum.
An entry under "mu" is a decimal number indicating the reduced mass.
An entry under "threshold" is a decimal number indicating a finite threshold
or the string "inf" if the potential is confining.
The ordering of the columns is irrelevant, but it must obviously be consistent between rows.
The typical structure of a minimal channels' CSV file is therefore:

BOF  
l, threshold, mu  
l1, T1, mu1  
l2, T2, mu2  
...  
lN, TN, muN  
EOF

For the user's convenience, additional information about the channels may be specified by adding more columns.
For instance, the user may want to include a column named "s" indicating the spin for each channel, or a column named "channel" providing a label that is more informative than just the integer $i$.
Such additional columns will be read by SPARSE but will not play any role in the calculation.

The positions of the nodes the values of the numerical potential must be stored in a CSV file with $M$ lines and **NO header**.
Each line must contain exactly $N^2 + 1$ entries.
The first entry is the node value $r$.
The remaining $N^2$ entries are the values of the flattened potential matrix at the given node.
SPARSE assumes that the potential is flattened in row-major (C-style) order
(this is irrelevant if the potential matrix is symmetric).
The typical structure of the potential's CSV file is therefore:

BOF  
r1, V11(r1), V12(r1), ..., V1N(r1), V21(r1), V22(r1), ..., V2N(r1), ..., VN1(r1), VN2(r1), ..., VNN(r1)  
r2, V11(r2), V12(r2), ..., V1N(r2), V21(r2), V22(r2), ..., V2N(r2), ..., VN1(r2), VN2(r2), ..., VNN(r2)  
...  
rM, V11(rM), V12(rM), ..., V1N(rM), V21(rM), V22(rM), ..., V2N(rM), ..., VN1(rN), VN2(rN), ..., VNN(rM)  
EOF

### An important note about units

Below is a dimensional analysis of the inputs.

**Columns of channels.csv:**  
(l) dimensionless (it is an integer)  
(threshold) [Energy]  
(mu) [Mass]  

**Columns of potential.csv:**  
(1) [Length]  
(2, 3, ..., $N^2 + 1$) [Energy]  

The SPARSE algorithm uses natural units, in which $\hbar=c=1$ and therefore $[\text{Mass}]=[\text{Energy}]$ and $[\text{Length}]=[\text{Energy}]^{-1}$.
There is only one independent dimension, typically chosen between [Length] in units of fm (femtometers) and [Energy] in units of eV (electron Volts).
The SPARSE algorithm is agnostic to the user's choice of dimension/unit, as long as it is consistent with the prescription $\hbar=c=1$.
The conversion between length-based and energy-based units is easily achieved by means of the formula $\hbar c = 1 = 197.327$ MeV fm.

## Repository structure

The SPARSE repository contains the following files:

*sparse.py*: the SPARSE module itself, make sure that it is accessible from the working directory in which the CSV files are located.

*utils.py*: a utility module containing handy functions for the further analysis of numerical results obtained using SPARSE.

*example_potential.py*: a Python script that automatically calculates an example potential matrix and writes it into CSV files compatible with SPARSE.

*SPARSE_tutorial.ipynb*: an IPython notebook tutorial for SPARSE, check it out!

*requirements.txt*: list of package requirements for the SPARSE module and its utility functions.
