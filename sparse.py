# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 12:13:03 2025

@author: Roberto Bruschini
"""
import warnings
import multiprocessing
import numpy as np
import pandas as pd
from scipy.linalg import solve_banded, inv
from scipy.integrate import trapezoid
from scipy.sparse import dia_array
from scipy.sparse.linalg import eigsh

# Import potential metadata from file: channels.csv
channels = pd.read_csv('channels.csv',
                       usecols=['channel', 'l', 'threshold', 'mu'],
                       dtype={
                           'channel': str,
                           'l': np.int64,
                           'threshold': np.float64,
                           'mu': np.float64,
                           })
channels.rename(index=lambda x: x + 1, inplace=True)

# Import potential matrix from file: potential.csv
potential = pd.read_csv('potential.csv',
                        index_col=0,
                        names=pd.MultiIndex.from_product(2 * [channels.index]),
                        dtype=np.float64)
potential.columns.set_names(['row', 'column'], inplace=True)

# Set up global variables from potential metadata.
l = channels.l.to_numpy()
threshold = channels.threshold.to_numpy()
mu = channels.mu.to_numpy()
r = potential.index.to_numpy()

# Check that the coordinates are positive, equally spaced, and start from 0.
dr = r[0]
assert dr > 0 and np.allclose(np.diff(r), dr), 'Malformed coordinate space.'

# Reshape the potential.
# Will result in an error if input potential is misshaped.
n = len(channels)
m = len(potential)
pot = potential.to_numpy().reshape(m, n, n)

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
# The kinetic energy matrix has only three nonzero diagonals.
# They add to the Hamiltonian's uppermost, central, and lowermost diagonals.
kinetic_off_diag = np.tile(-1 / (2 * mu * dr ** 2), m - 1)
hamiltonian[0] = np.pad(kinetic_off_diag, (n, 0))
hamiltonian[-1] = np.pad(kinetic_off_diag, (0, n))

# Establish the scattering channels as those with a finite threshold.
scattering = threshold < np.inf

# Establish the energy values excluded from calculating the amplitudes.
# The limits are structured as a (2 + N, 2)-array where N is the number
# of scattering channels. Each row of the array is a pair of values.
# Energies within any such pair of values is excluded.
elims = np.empty((2 + np.count_nonzero(scattering), 2))

# Overall lower and upper limits are set as (-inf, emin) and (emax, +inf).
# The lower limit is the lowest finite threshold.
emin = threshold.min()
elims[0] = (-np.inf, emin)
# Calculate the maximum momentum based on the discretization distance.
# The number in the denominator should be bigger than 1.
pmax = np.pi / (dr * 500)  # <- change "500" if needed
# The upper limit is dictated by the maximum scattering momentum or
# the potentials for non-scattering channels, whichever is smaller.
emax_scattering = pmax**2 / (2 * mu[scattering]) + threshold[scattering]
emax_bound = pot[-1].diagonal()[~scattering]
emax = min(*emax_scattering, *emax_bound)
elims[1] = (emax, np.inf)

# Establish exclusion zones for energies too close to finite thresholds.
# Energies below threshold are limited by the maximum binding momentum.
# Calculate the maximum binding momentum based on the maximum distance.
# The number in the numerator should be bigger than 1.
bound_p_max = 10 / r[-1]  # change "10" if needed
# Energies above threshold are limited by the minimum scattering momentum.
# Calculate the radius of the potential for scattering channels.
# The potential is considered 0 if its value is less than atol.
pot_scattering = pot[np.ix_(range(m), scattering, scattering)]
pot_zero_ref = np.diag(threshold[scattering])
pot_is_zero = np.isclose(pot_scattering, pot_zero_ref,
                         atol=1e-8).all(axis=(1,2))  # <- change atol if needed
# The potential radius is defined as the largest value of r
# for which the corresponding value of pot_is_flat is False.
# Check that the potential radius is within the coordinate space.
assert pot_is_zero[-1], 'Potential is nonzero at maximum radius.'
# Assign False to the pot_is_flat[0] to ensure that
# the potential radius is not smaller than r[0].
pot_is_zero[0] = False
# Calculate the potential radius.
r_pot = r[~pot_is_zero][-1]
# Calculate the minimum scattering momentum based on the potential radius.
free_p_min_pot = np.pi / (r[-1] - r_pot)
# Calculate the minimum scattering momentum from the condition that the
# analytic scattering states are approximately sine and cosine functions.
# The number in the numerator should be bigger than 1.
free_p_min_l = (10 * l[scattering] + np.pi) / r[-1] # change "10" if needed
# The minimum scattering momentum is the largest between those two.
free_p_min = np.maximum(free_p_min_pot, free_p_min_l)
elims[2:, 0] = threshold[scattering] \
    - bound_p_max ** 2 / (2 * mu[scattering])
elims[2:, 1] = threshold[scattering] \
    + free_p_min ** 2 / (2 * mu[scattering])


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
    if np.logical_and(energy > elims[:, 0], energy < elims[:, 1]).any():
        raise ValueError('Invalid input energy.')
    is_open = energy > threshold
    o = np.count_nonzero(is_open)
    l_open = l[is_open]
    t_open = threshold[is_open]
    mu_open = mu[is_open]
    p = np.sqrt(2 * mu_open * (energy - t_open))
    ab = hamiltonian.copy()
    ab[n] -= energy
    b = np.zeros((n * m, o))
    b[-n:][is_open] = np.diag(1 / (2 * mu_open * dr ** 2))
    vec = solve_banded((n, n), ab, b, overwrite_ab=True,
                       overwrite_b=True, check_finite=False)
    sol = vec.reshape(m, n, o)[:, is_open]
    jmatrix = np.empty((o, o))
    hmatrix = np.empty_like(jmatrix)
    for i in range(o):
        dx = dr * p[i]
        j = (np.pi / dx).round().astype(int) + 1
        x = r[-j:] * p[i] - l_open[i] * np.pi / 2
        y = sol[-j:, i].T
        normfactor = np.sqrt(p[i] / mu_open[i])
        jmatrix[i] = trapezoid(y * np.cos(x), dx=dx) * normfactor
        hmatrix[i] = trapezoid(y * np.sin(x), dx=dx) * normfactor
    kmatrix_raw = jmatrix @ inv(hmatrix, overwrite_a=True, check_finite=False)
    kmatrix = (kmatrix_raw + kmatrix_raw.T) / 2
    if not np.allclose(kmatrix_raw, kmatrix, rtol=rtol):
        warnings.warn(
            'Asymmetry tolerance exceeded.',
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
    below = energies < elims[:, 0, np.newaxis]
    above = energies > elims[:, 1, np.newaxis]
    outside = np.logical_or(below, above)
    energies = energies[outside.all(axis=0)]
    if processes == 1 or __name__ == "__main__":
        kmatrices = [k_matrix(e) for e in energies]
    else:
        with multiprocessing.Pool(processes) as pool:
            results = pool.map_async(k_matrix, energies)
            kmatrices = results.get()
    k_array = np.empty((len(energies), n, n))
    for i, k in enumerate(kmatrices):
        is_open = energies[i] > threshold
        k_array[i][np.ix_(is_open, is_open)] = k
        k_array[i][np.ix_(is_open, ~is_open)] = np.nan
        k_array[i][~is_open] = np.nan
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
    if energy_guess < e_bound_max:
        raise ValueError('Invalid energy guess.')
    offsets = np.arange(n, -n - 1, -1)
    hb = dia_array((hamiltonian, offsets), shape=(n * m, n * m))
    eigenvalues, eigenvectors = eigsh(hb, k=n_states, sigma=energy_guess)
    if any(above := eigenvalues > e_bound_max):
        warnings.warn(
            f'Removing {np.count_nonzero(above)} states above {e_bound_max}',
            RuntimeWarning, stacklevel=2)
        eigenvalues = eigenvalues[~above]
        eigenvectors = eigenvectors[..., ~above]
        n_states -= sum(above)
    if any(eigenvalues > min(elims[1:, 0])):
        warnings.warn(f'States above {min(elims[1:, 0])} may be distorted',
                      RuntimeWarning, stacklevel=2)
    norm_vec = eigenvectors.reshape(m, n, n_states) / np.sqrt(dr)
    r_closure = np.insert(r, [0, m], [0, r[-1] + dr])
    wavefuncs = np.insert(norm_vec, [0, m], 0, axis=0)
    wave_functions = np.empty(n_states, dtype=object)
    for i in range(n_states):
        wave_functions[i] = pd.DataFrame(wavefuncs[..., i],
                                         index=r_closure,
                                         columns=channels.index)
    return eigenvalues, wave_functions
