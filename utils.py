# -*- coding: utf-8 -*-
"""
Created on Wed May 28 15:11:17 2025

@author: rober
"""
import numpy as np
import pandas as pd
from scipy.linalg import inv, det
from scipy.interpolate import AAA


def poles(kmatrix_df, rtol=1e-2, clean_up_tol=1e-13):
    """
    Calculate scattering poles by extrapolating input K-matrices.

    Parameters
    ----------
    kmatrix_df : DataFrame
        Pandas DataFrame containing flattened K-matrices for various energies.
        This is typically an output of the function k_matrices from the SPARSE
        module.
    rtol : float, optional
        Relative tolerance for the AAA rational approximation algorithm.
        If too small, spurious poles may appear in the rational approximation.
        See the documentation of scipy.interpolate.AAA for furhter information.
        The default is 1e-3.
    clean_up_tol : float, optional
        Threshold for automatic clean up of spurious poles. See the
        documentation of scipy.interpolate.AAA for further information.
        The default is 1e-13.

    Returns
    -------
    (N,) ndarray
        The extrapolated N scattering poles in the complex energy plane closest
        to the physical axis, ordered by increasing real part.

    """
    x = kmatrix_df.index.to_numpy()
    y = np.empty(len(x), dtype=np.complex128)
    for i, kmat in enumerate(kmatrix_df.to_numpy()):
        kflat = kmat[~np.isnan(kmat)]
        n = int(np.sqrt(len(kflat)))
        k = kflat.reshape(n, n)
        y[i] = det(np.eye(n) - 1j * k, overwrite_a=True, check_finite=False)
    r = AAA(x, y, rtol=rtol, clean_up_tol=clean_up_tol)
    return np.sort_complex(r.roots())


def amplitudes(kmatrix_df):
    """
    Calculate scattering amplitudes from the input K-matrices.

    Parameters
    ----------
    kmatrix_df : DataFrame
        Pandas DataFrame containing flattened K-matrices for various energies.
        This is typically an output of the function k_matrices from the SPARSE
        module.

    Returns
    -------
    DataFrame
        Pandas DataFrame containing the flattened scattering amplitudes.
        Matrix elements corresponding to closed channels are set to NaN.

    """
    amplitudes = np.empty(kmatrix_df.shape, dtype=np.complex128)
    for i, kmat in enumerate(kmatrix_df.to_numpy()):
        kflat = kmat[~np.isnan(kmat)]
        n = int(np.sqrt(len(kflat)))
        k = kflat.reshape(n, n)
        t = k @ inv(np.eye(n) - 1j * k, overwrite_a=True, check_finite=False)
        amplitudes[i, ~np.isnan(kmat)] = t.flatten()
        amplitudes[i, np.isnan(kmat)] = np.nan + 1j * np.nan
    return pd.DataFrame(amplitudes,
                        index=kmatrix_df.index,
                        columns=kmatrix_df.columns)


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
