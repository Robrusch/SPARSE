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
from scipy.special import spherical_jn, spherical_yn
from scipy.optimize import brentq
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
                        names=pd.MultiIndex.from_product(
                            2 * [channels.index.astype(str)]
                            ),
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
# The energy must be smaller than 10% of the upper limit
emax = np.concatenate((emax_scattering, emax_bound)).min() / 10

# Confirm tha the potential for scattering channels is negligible at
# the maximum integration radius. Excluding diagonal elements for
# confining channels with infinite thresholds.
# Then, calculate the potential radius.
pot_is_thresh = np.isclose(pot, np.diag(threshold)[np.newaxis])
pot_is_thresh[:, ~scattering, ~scattering] = True
assert pot_is_thresh[-1].all(),\
    'Potential matrix at maximum radius is significantly different from' \
        'threshold matrix, check your channels and potential.'
pot_radius_ix = np.argwhere(np.logical_not(pot_is_thresh).any(axis=(1,2)))[-1, 0]

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


def k_matrix(energy):
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
    assert energy > emin, f'Energies must be higher than {emin}. '\
        '(No scattering for energies below the lowest threshold).'
    assert energy < emax, f'Energies must be lower than {emax}. '\
        '(Increase the maximum radius and/or decrease the '\
            'discretization step to increase the maximum energy).'
    is_open = energy > threshold
    o = np.count_nonzero(is_open)    
    p = np.sqrt(2 * mu[is_open] * (energy - threshold[is_open]))
    norm = np.sqrt(2 * mu[is_open] / (np.pi * p))
    boundary = np.zeros((n, o))
    boundary[is_open] = np.diag(norm)
    wavefuncs = wavefunctions(energy, boundary)
    u = wavefuncs[pot_radius_ix:, is_open] / norm[..., np.newaxis]
    li = l[np.newaxis, is_open]
    z = np.outer(r_space[pot_radius_ix:], p)
    s = np.expand_dims(z * spherical_jn(li, z), axis=2)
    c = np.expand_dims(- z * spherical_yn(li, z), axis=2)
    alpha = np.sum(s ** 2, axis=0)
    beta = np.sum(c ** 2, axis=0)
    gamma = np.sum(s * c, axis=0)
    delta = alpha * beta - gamma ** 2
    a = np.sum(u * s, axis=0)
    b = np.sum(u * c, axis=0)
    x = (beta * a - gamma * b) / delta
    y = (alpha * b - gamma * a) / delta
    kmatrix_asym = y @ inv(x, overwrite_a=True, check_finite=False)
    kmatrix = (kmatrix_asym + kmatrix_asym.T) / 2
    if not np.allclose(kmatrix_asym, kmatrix):
        warnings.warn(
            f'K-matrix asymmetry detected for E={energy}.',
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
    above_emin = energies > emin
    below_emax = energies < emax
    admitted = np.logical_and(above_emin, below_emax)
    if not admitted.all():
        excluded = np.count_nonzero(~admitted)
        warning = f'Removing {excluded} invalid energy value(s) from input.'
        if not above_emin.all():
            warning += f'Energies must be higher than {emin}. '\
                '(No scattering for energies below the lowest threshold).'
        if not below_emax.all():
            warning += f'Energies must be lower than {emax}. '\
                '(Increase the maximum radius and/or decrease the '\
                    'discretization step to increase the maximum energy).'
        warnings.warn(warning, stacklevel=2)
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


def k_matrix_pole(e1, e2, de, epsilon=None, points=100, deg=10, processes=1):
    """
    Calculate a K-matrix pole and its residues in a given energy interval.
    
    Parameters
    ----------
    e1: float
        The lower extreme of the energy interval.
    
    e2: float
        The higher extreme of the energy interval.
    
    de: float
        Interval around the pole position used to calculate the residue.
    
    epsilon: float, optional
        Exclude values closer than epsilon to the pole position when calculating
        the residue. The default epsilon=None correspond to using epsilon=de.
    
    points: int, optional
        Number of discrete energy points used when calculating the residue of
        the K-matrix. Default is 100.
    
    deg: int, optional
        Maximum number of the polynomial used in the interpolation around the
        pole position, used to calculate the pole residue. Default is 10.
    
    processes: int, optional
        How many processes to use to calculate the pole residue. Default is 1,
        which does not invoke multiprocessing.
    
    Returns
    -------
        float, float, (,N) ndarray
        The nominal mass, nominal width, and nominal couplings of the pole.
        
    """
    assert e1 < e2, 'e1 must be smaller than e2.'
    threshold_between = np.logical_and(threshold >= e1, threshold <= e2)
    assert not any(threshold_between), 'Energy interval includes threshold(s)'\
        f'at {threshold[threshold_between]}'
    assert e1 > emin, f'Energy interval must be higher than {emin}. '\
        '(No scattering for energies below the lowest threshold).'
    assert e2 < emax, f'Energy interval must be lower than {emax}. '\
        '(Increase the maximum radius and/or decrease the '\
            'discretization step to increase the maximum energy).'
    mass = brentq(lambda e: 1 / np.trace(k_matrix(e)), e1, e2)
    assert abs(np.trace(k_matrix(mass))) > 1, 'Found zero of K-matrix trace '\
        f'at {mass}.'
    if epsilon is None:
        epsilon = de
    eminus = np.linspace(mass - epsilon / 2 - de * (points - 1),
                         mass - epsilon / 2,
                         points)
    threshold_below = max(threshold[threshold < mass])
    eminus = eminus[eminus > threshold_below]
    eplus = np.linspace(mass + epsilon / 2,
                        mass + epsilon / 2 + de * (points - 1),
                        points)
    try:
        threshold_above = min(threshold[threshold > mass])
    except ValueError:
        threshold_above = np.inf
    eplus = eplus[eplus < threshold_above]
    en = np.concat((eminus, eplus))
    kmat = k_matrices(en, processes)
    o = int(np.sqrt(len(kmat.columns)))
    k = kmat.to_numpy().reshape(-1, o, o).transpose(1,2,0)
    x = kmat.index.to_numpy() - mass
    y = k * x
    residues = np.empty((o, o))
    for i in range(o):
        for j in range(o):        
            poly = np.polynomial.Polynomial.fit(x, y[i, j], deg)
            residues[i, j] = poly.convert().coef[0]
    res_diag = np.diagonal(residues)
    assert np.all(res_diag <= 0),\
        f'Non-resonant poles detected at {mass}.'\
            f' The diagonal of the residue matrix is {res_diag}.'
    res_trace = np.trace(residues)
    width = -2 * res_trace
    couplings = np.sign(-residues[0]) *\
        np.sqrt(res_diag / res_trace)
    return mass, width, couplings


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
    norm_vec = eigenvectors.reshape(m, n, n_states) / np.sqrt(dr)
    wavefuncs = np.insert(norm_vec, [0, m], 0, axis=0)
    wave_functions = np.empty(n_states, dtype=object)
    for i in range(n_states):
        wave_functions[i] = pd.DataFrame(wavefuncs[..., i],
                                         index=r_space,
                                         columns=channels.index)
    return eigenvalues, wave_functions
