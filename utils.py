# -*- coding: utf-8 -*-
"""
Created on Wed May 28 15:11:17 2025

@author: Roberto Bruschini
"""
import numpy as np
import pandas as pd
import warnings
from scipy.linalg import inv, det
from scipy.interpolate import AAA


def poles(k_matrix_df, **kwargs):
    """
    Calculate scattering poles by extrapolating input K-matrices.

    Parameters
    ----------
    k_matrix_df : DataFrame
        Pandas DataFrame containing flattened K-matrices for various energies.
        This is typically an output of the function k_matrices from the SPARSE
        module.
    **kwargs: optional
        Keyword arguments the function scipy.interpolate.AAA. See its
        documentation of  for further information.

    Returns
    -------
    t_matrix_poles : ndarray
        The extrapolated scattering poles in the complex energy plane,
        ordered by increasing real part.
    k_matrix_poles : ndarray
        The extrapolated poles of the K-matrix.
        These poles are not useful per se, but rather as an indication of
        whether or not there are spurious poles in the extrapolation.
        Adjust the keyword arguments until there is the correct number and
        position of K-matrix poles.

    """
    warnings.warn("This function is deprecated, use instead k_matrix_poles(). The extrapolation of the T-matrix poles for complex energies is numerically unstable. The function poles() will be removed in a future release.", DeprecationWarning, stacklevel=2)
    x = k_matrix_df.index.to_numpy()
    y = np.empty(len(x), dtype=np.complex128)
    for i, kmat in enumerate(k_matrix_df.to_numpy()):
        kflat = kmat[~np.isnan(kmat)]
        n = int(np.sqrt(len(kflat)))
        k = kflat.reshape(n, n)
        y[i] = det(np.eye(n) - 1j * k, overwrite_a=True, check_finite=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        r = AAA(x, y, **kwargs)
    k_matrix_poles = np.sort_complex(r.poles())
    t_matrix_poles = np.sort_complex(r.roots())
    return t_matrix_poles, k_matrix_poles


def k_matrix_poles(k_matrix, n_poles):
    kmat = k_matrix.dropna(axis=1)
    n = int(np.sqrt(len(kmat.columns)))
    poles = np.empty((n_poles, n, n))
    residues = np.empty_like(poles)
    levels = kmat.columns.remove_unused_levels().levels[0]
    for ni, i in enumerate(levels):
        for nj, j in enumerate(levels):
            k = kmat[i, j].dropna()
            x = k.index.to_numpy()
            y = k.to_numpy()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                r = AAA(x, y, max_terms=n_poles+1)
            assert np.isreal(r.poles).all() and np.isreal(r.residues).all()
            order = np.argsort(r.poles().real)
            poles[:, ni, nj] = r.poles().real[order]
            residues[:, ni, nj] = r.residues().real[order]
    res_diag = np.diagonal(residues, axis1=1, axis2=2)
    assert np.all(res_diag < 0)
    masses = poles[:, 0, 0]
    assert np.allclose(poles, masses[:, np.newaxis, np.newaxis])
    res_trace = np.sum(res_diag, axis=1)
    widths = -2 * res_trace
    couplings = np.sqrt(res_diag / res_trace)
    couplings[:, 1:] *= -np.sign(residues[:, 1:, 0])
    assert np.allclose(couplings[:,np.newaxis] * couplings[..., np.newaxis], residues / res_trace)
    return masses, widths, couplings


def amplitudes(k_matrix_df):
    """
    Calculate scattering amplitudes from the input K-matrices.

    Parameters
    ----------
    k_matrix_df : DataFrame
        Pandas DataFrame containing flattened K-matrices for various energies.
        This is typically an output of the function k_matrices from the SPARSE
        module.

    Returns
    -------
    DataFrame
        Pandas DataFrame containing the flattened scattering amplitudes.
        Matrix elements corresponding to closed channels are set to NaN.

    """
    amplitudes = np.full(k_matrix_df.shape, np.nan + 1j * np.nan)
    for i, kmat in enumerate(k_matrix_df.to_numpy()):
        kflat = kmat[~np.isnan(kmat)]
        n = int(np.sqrt(len(kflat)))
        k = kflat.reshape(n, n)
        t = k @ inv(np.eye(n) - 1j * k, overwrite_a=True, check_finite=False)
        amplitudes[i, ~np.isnan(kmat)] = t.flatten()
    return pd.DataFrame(amplitudes,
                        index=k_matrix_df.index,
                        columns=k_matrix_df.columns)


def composition(wavefunc_df):
    """
    Calculate the probabilities of a bound state in the various channels.

    Parameters
    ----------
    wavefunc_df : DataFrame
        Pandas DataFrames containing the wave function.
        This is typically an output of the function bound_states from
        the SPARSE module.

    Returns
    -------
    Series
        Pandas series containing the probabilities in decreasing order.

    """
    psi_squared = np.square(wavefunc_df.to_numpy())
    prob = np.trapz(psi_squared, wavefunc_df.index.to_numpy(), axis=0)
    order = np.argsort(prob)[::-1]
    return pd.Series(prob[order], wavefunc_df.columns[order])
