# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 12:13:03 2025

@author: Roberto Bruschini
"""
import warnings
import multiprocessing
import math
import numpy as np
import pandas as pd
from scipy.linalg import solve_banded, inv
from scipy.integrate import simpson
from scipy.sparse import dia_array
from scipy.sparse.linalg import eigsh

# Import channel description from file: channels.csv
channels = pd.read_csv('channels.csv',
                       dtype={
                           'l': int,
                           'threshold': float,
                           'mu': float,
                           })
channels.rename(index=lambda x: x + 1, inplace=True)

# Import coordinate space and potential matrix from file: potential.csv
potential = pd.read_csv('potential.csv',
                        index_col=0,
                        names=pd.MultiIndex.from_product(2 * [channels.index.astype(str)]),
                        dtype=float)
potential.columns.set_names(['row', 'column'], inplace=True)

# Set up global variables from channel description
for val in ['l', 'threshold', 'mu']:
    assert val in channels.columns, f'Missing column "{val}" in channels.'
l = channels.l.to_numpy()
threshold = channels.threshold.to_numpy()
mu = channels.mu.to_numpy()

# Reshape the potential.
n = len(channels)
m, nsquared = potential.shape
assert nsquared == n ** 2, 'Shape mismatch between channels and potential. ' \
    f'Number of channels squared ({n **2}) does not coincide with ' \
        f'length of flattened potential matrix ({nsquared}).'
pot = potential.to_numpy().reshape(m, n, n)

# Set up the coordinate space.
r = potential.index.to_numpy()
dr = r[0]
assert dr > 0 and np.allclose(np.diff(r), dr), \
    'Malformed coordinate space, check your potential.'
r_space = np.insert(r, [0, m], [0, r[-1] + dr])

# Establish the scattering channels as those with a finite threshold.
scattering = threshold < np.inf

# Set overall lower and upper energy limits.
# The lower limit is the lowest finite threshold.
emin = threshold.min()
# Calculate the maximum momentum based on the discretization distance.
pmax = np.pi / dr
# The upper limit is dictated by the maximum scattering momentum or
# the potentials for non-scattering channels, whichever is smaller.
emax_scattering = pmax**2 / (2 * mu[scattering]) + threshold[scattering]
emax_bound = pot[-1].diagonal()[~scattering]
emax = np.concatenate((emax_scattering, emax_bound)).min()

# Establish exclusion zones for energies too close to finite thresholds.
# The limits consist of a pair of energies for each scattering channel.
# Energies within any such pair of values are excluded.
elims = np.empty((2, np.count_nonzero(scattering)))

# Energies below threshold are limited by the maximum binding momentum.
# Calculate the minimum binding momentum based on the maximum distance.
# The first number in the numerator should be bigger than 1.
bound_p_min = 10 / r_space[-1]  # <- may change "10"
elims[0] = threshold[scattering] \
    - bound_p_min ** 2 / (2 * mu[scattering])

# Energies above threshold are limited by the minimum scattering momentum.
# Calculate the radius of the potential for scattering channels.
# The difference between the potential and threshold matrices is considered
# zero if its value is less than atol.
thresh = np.diag(threshold)
pot_are_thresh = np.isclose(pot, thresh, atol=1e-8)  # <- may change atol
# Exclude diagonal elements for confining channels with infinite thresholds
# by setting the corresponding elements of pot_are_thresh to True.
pot_are_thresh[:, ~scattering, ~scattering] = True
pot_is_thresh = pot_are_thresh.all(axis=(1,2))
# The potential radius is defined as the largest value of r
# for which the corresponding value of pot_is_thresh is False.
# Check that the potential radius is within the coordinate space.
assert pot_is_thresh[-1], \
    'Potential matrix at maximum radius is significantly different from' \
        'threshold matrix, check your channels and potential.'
# Calculate the potential radius.
if all(pot_is_thresh):
    r_pot = 0
else:
    r_pot = r[~pot_is_thresh][-1]
# Calculate the minimum scattering momentum based on the potential radius.
free_p_min_pot = np.pi / (r_space[-1] - r_pot)
# Calculate the minimum scattering momentum from the condition that the
# analytic scattering states are approximately sine and cosine functions.
# The number in the numerator should be bigger than 1.
free_p_min_l = (10 * l[scattering] + np.pi) / r_space[-1] # <- may change "10"
# The minimum scattering momentum is the largest between those two.
free_p_min = np.maximum(free_p_min_pot, free_p_min_l)
elims[1] = threshold[scattering] \
    + free_p_min ** 2 / (2 * mu[scattering])

# Calculate the sparse Hamiltonian matrix.
# The Hamiltonian matrix is a real, square matrix of dimension (n * m)^2.
# It has only 2 * n + 1 nonzero diagonals with the main one at the center.
# The Hamiltonian is stored using the (padded) matrix diagonal ordered form.
# See the documentation of scipy.linalg.solve_banded for more information.
hamiltonian = np.empty((2 * n + 1, n * m))
# Calculate the main diagonal.
kinetic_main_diag = np.tile(1 / (mu * dr ** 2), m)
centrifugal_diag = np.outer(r ** -2, l * (l + 1) / (2 * mu)).flatten()
pot_main_diag = pot.diagonal(0, 1, 2).flatten()
hamiltonian[n] = kinetic_main_diag + centrifugal_diag + pot_main_diag
# Calculate the upper and lower diagonals.
for k in range(1, n):
    upp_diags = pot.diagonal(k, 1, 2)
    low_diags = pot.diagonal(-k, 1, 2)
    hamiltonian[n - k] = np.pad(upp_diags, ((0,0), (k, 0))).flatten()
    hamiltonian[n + k] = np.pad(low_diags, ((0,0), (0, k))).flatten()
kinetic_off_diag = np.tile(-1 / (2 * mu * dr ** 2), m - 1)
hamiltonian[0] = np.pad(kinetic_off_diag, (n, 0))
hamiltonian[-1] = np.pad(kinetic_off_diag, (0, n))


def wavefunctions(energy, boundary):
    """
    Computes the numerical wavefunctions with given boundary conditions
    at the maximum radius.

    Parameters
    ----------
    energy : float
        Input energy value.
    boundary : (N, O) ndarray
        An array containing the boundary conditions for the wavefunctions.
        The number N of rows must be the number of wavefunction channels.
        The number O of columns corresponds to the number of wavefunctions
        with the same energy returned by this function.

    Returns
    -------
    (M, N, O) ndarray
        The numerical wavefunctions with their boundary values.

    """
    ab = hamiltonian.copy()
    ab[n] -= energy
    o = boundary.shape[1]
    b = np.zeros((n * m, o))
    b[-n:] = boundary / (2 * mu[:, np.newaxis] * dr ** 2)
    sol = solve_banded((n, n), ab, b, overwrite_ab=True,
                       overwrite_b=True, check_finite=False)
    vec = sol.reshape(m, n, o)
    wavefuncs = np.insert(vec, [0, m], [np.zeros((n, o)), boundary], axis=0)
    return wavefuncs


def k_matrix(energy, rtol=1e-2):
    """
    Compute the K-matrix at a given energy.

    Parameters
    ----------
    energy : float
        Input energy value, must be outside all regions excluded by elims.
    rtol : float, optional
        Relative asymmetry tolerance for the K-matrix. The default is 1%.

    Returns
    -------
    (N, N) ndarray
        The K-matrix for the N open channels at the input energy.

    """
    if np.logical_and(energy > elims[0], energy < elims[1]).any() \
        or energy < emin or energy > emax:
        raise ValueError('Invalid input energy.')
    is_open = energy > threshold
    o = np.count_nonzero(is_open)
    boundary = np.zeros((n, o))
    boundary[is_open] = np.eye(o)
    wavefuncs = wavefunctions(energy, boundary)
    y = np.empty((o, o))
    x = np.empty_like(y)
    for i, j in enumerate(np.argwhere(is_open).flatten()):
        p = np.sqrt(2 * mu[j] * (energy - threshold[j]))
        mult = np.sqrt(2 * np.pi * p**3 / mu[j])
        r_min = max(r_pot, 10 * l[j] / p)
        lim = math.ceil((r_space[-1] - r_min) / dr)
        z = r_space[-lim:] * p - l[j] * np.pi / 2
        w = wavefuncs[-lim:, j].T
        s = mult * simpson(w * np.sin(z), dx=dr)
        c = mult * simpson(w * np.cos(z), dx=dr)
        alpha = z[-1] - z[0]
        beta = (np.sin(2 * z[-1]) - np.sin(2 * z[0])) / 2
        gamma = (np.cos(2 * z[-1]) - np.cos(2 * z[0])) / 2
        eta = alpha ** 2 - beta ** 2 - gamma ** 2
        x[i] = ((alpha + beta) * s + gamma * c) / eta
        y[i] = (gamma * s + (alpha - beta) * c) / eta
    kmatrix_asym = y @ inv(x, overwrite_a=True, check_finite=False)
    kmatrix = (kmatrix_asym + kmatrix_asym.T) / 2
    if not np.allclose(kmatrix_asym, kmatrix, rtol=rtol):
        warnings.warn(
            f'Asymmetry tolerance exceeded for E={energy}.',
            RuntimeWarning, stacklevel=2)
    return kmatrix


def k_matrices(energies, processes=1):
    """
    Compute the K-matrices for an array of energies.

    Parameters
    ----------
    energies : (N,) array_like
        Input energy values. Values that fall within any region excluded by
        elims are automatically discarded.
    processes : int, optional
        Number of parallel processes used for the task. Multiprocessing works
        only if this function is loaded via an import statement. Otherwise this
        option is ignored. The default is 1.

    Returns
    -------
    DataFrame
        Pandas DataFrame containing the flattened K-matrices.
        Matrix elements corresponding to closed channels are set to NaN.

    """
    if type(energies) != np.ndarray:
        raise TypeError('Energies must be provided as a numpy array.')
    below = energies < elims[0, ..., np.newaxis]
    above = energies > elims[1, ..., np.newaxis]
    away = np.logical_or(below, above).all(axis=0)
    between = np.logical_and(energies > emin, energies < emax)
    admitted = np.logical_and(away, between)
    if not admitted.all():
        excluded = np.count_nonzero(~admitted)
        out_of_bounds = np.count_nonzero(~between)
        close_to_threshold = np.count_nonzero(~away)
        warnings.warn(f'Removing {excluded} energy values from input '
                      f'({close_to_threshold} too close to threshold, '
                      f'{out_of_bounds} outside energy limits)',
                      stacklevel=2)
        if not admitted.any():
            raise ValueError('There is no valid energy in input.')
    energies = energies[admitted]
    if processes == 1 or __name__ == "__main__":
        kmatrices = list(map(k_matrix, energies))
    else:
        with multiprocessing.Pool(processes) as pool:
            kmatrices = pool.map(k_matrix, energies, chunksize=1)
    k_array = np.full((len(energies), n, n), np.nan)
    for i, k in enumerate(kmatrices):
        is_open = energies[i] > threshold
        k_array[i][np.ix_(is_open, is_open)] = k
    k_flat = k_array.reshape(len(energies), -1)
    kdf = pd.DataFrame(k_flat, index=energies, columns=potential.columns)
    return kdf.dropna(axis=1, how='all')


def bound_states(n_states, energy_guess):
    """
    Calculate bound states for energies below the lowest finite threshold.

    Parameters
    ----------
    n_states : int
        Number of expected bound states.
    energy_guess : float
        Energy guess for the bound states. The function will try to calculate
        n_states eigenvectors with eigenvalue closest to energy_guess.

    Returns
    -------
    eigenvalues : (N,) ndarray of floats
        The bound-state energies.
    wave_functions : (N,) ndarray of DataFrames
        Array of pandas DataFrames for the reduced radial wave functions.

    """
    e_bound_max = min(emin, emax)
    if energy_guess > e_bound_max:
        raise ValueError('Invalid energy guess.')
    offsets = np.arange(n, -n - 1, -1)
    hb = dia_array((hamiltonian, offsets), shape=(n * m, n * m))
    eigenvalues, eigenvectors = eigsh(hb, k=n_states, sigma=energy_guess)
    if any(above := eigenvalues > e_bound_max):
        warnings.warn(
            f'Removing {np.count_nonzero(above)} states above {e_bound_max}',
            stacklevel=2)
        eigenvalues = eigenvalues[~above]
        eigenvectors = eigenvectors[..., ~above]
    if len(eigenvalues) == 0:
        raise RuntimeError('No bound states found')
    if any(scattering):
        if any(eigenvalues > elims[0].min()):
            warnings.warn(f'States above {elims[0].min()} may be distorted',
                          RuntimeWarning, stacklevel=2)
    norm_vec = eigenvectors.reshape(m, n, n_states) / np.sqrt(dr)
    wavefuncs = np.insert(norm_vec, [0, m], 0, axis=0)
    wave_functions = np.empty(n_states, dtype=object)
    for i in range(n_states):
        wave_functions[i] = pd.DataFrame(wavefuncs[..., i],
                                         index=r_space,
                                         columns=channels.index)
    return eigenvalues, wave_functions
