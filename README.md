# SPARSE: Scattering Poles and Amplitudes from Radial Schrödinger Equations

## Introduction

This algorithm calculates scattering amplitudes from a two-body, multi-channel, radial Schrödinger equation in the form:

$$\left(-\frac{1}{2 \mu} \frac{\mathrm{d}^2}{\mathrm{d}r^2} + \frac{L (L + 1)}{2 \mu r^2} + V(r) \right) \psi(r) = E \psi(r)$$

where $r$ is the relative distance between the two degrees of freedom,
$\psi(r)=(\psi_1(r),\psi_2(r),\dots,\psi_N(r))$ is a wave function with N channels,
$\mu = \mathrm{diag}(\mu_1, \mu_2, \dots, \mu_N)$ is a diagonal reduced-mass matrix,
$L=\mathrm{diag}(L_1, L_2, \dots, L_N)$ is a diagonal orbital-angular-momentum matrix,
and $V(r)$ is a $N \times N$ potential matrix with both diagonal and off-diagonal entries that depends on $r$.
Note that we use natural units: $\hbar=c=1$.

We define the threshold matrix as the limit of the potential matrix for $r\to\infty$: $T=\lim_{r\to\infty}V(r)$.
This module focuses on cases where all limits are well defined,
the threshold matrix is diagonal, $T=\mathrm{diag}(T_1,T_2,\dots,T_N)$, and each threshold is either a finite number or $+\infty$.
The Schrödinger equation then admits scattering solutions for energies larger than the lowest threshold, $E \geq \min T$.

SPARSE solves the Schrödinger numerically using the finite difference method by sampling the continuous distance $r$ on a discrete
lattice of point separated by a constant distance $\mathrm{d}r$. The lattice begins at $r_\text{min}=0$, contains $M$ equally-spaced points in its interior
$r_i=i ~ \mathrm{d}r$ with $i=1,\dots,M$, and ends at $r_\text{max}=(M+1) \mathrm{d}r$.
The wave function $\psi$ with N channels is treated as a numerical array containing its values at each of the M interior nodes of $r$.
(The boundary points are treated in a different manner.)
It is an array with $N \times M$ entries. Similarly, the potential matrix $V$ is treated as an array with $N \times N \times M$ entries, corresponding to
each entry of the potential matrix evaluated at each interal node.

## Setup

The numerical Schrödinger equation is completely specified by 4 quantities:  
1. the channels' orbital angular momenta;
2. the channels' thresholds;
3. the channels' reduced masses;
4. the numerical potential matrix.

The channels' specifics (name, orbital angular momenta, thresholds, reduced masses) must be stored in a CSV with 1 line of header and $N$ additional lines below.
Each line must contain 4 entries.
The header must contain the following strings: channel, l, threshold, mu.
The following lines must contain the corresponding values.
The ordering of the columns is irrelevant, but must obviously be consistent with the tabular format.
The typical structure of the channels' CSV file is therefore:

BOF  
channel, l, threshold, mu  
Name1, l1, T1, mu1,  
Name2, l2, T2, mu2,  
...  
NameN, lN, TN, muN  
EOF  

The nodes positions and the numerical potential must be stored in a CSV file with $M$ lines and **NO header**.
Each line must contain $1 + N^2$ entries.
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

## Repository structure

The SPARSE repository contains the following files:

*sparse.py*: the SPARSE module itself, make sure that it is accessible from the working directory in which the CSV files are located.

*utils.py*: a utility module containing handy functions for the further analysis of numerical results obtained using SPARSE.

*example.py*: a script that automatically calculates and writes two examples of working CSV files.

*SPARSE-tutorial.ipynb*: an IPython notebook tutorial for SPARSE.
