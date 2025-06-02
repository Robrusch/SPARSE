# -*- coding: utf-8 -*-
"""
Created on Wed May 28 15:11:17 2025

@author: rober
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
    amplitudes = np.empty(k_matrix_df.shape, dtype=np.complex128)
    for i, kmat in enumerate(k_matrix_df.to_numpy()):
        kflat = kmat[~np.isnan(kmat)]
        n = int(np.sqrt(len(kflat)))
        k = kflat.reshape(n, n)
        t = k @ inv(np.eye(n) - 1j * k, overwrite_a=True, check_finite=False)
        amplitudes[i, ~np.isnan(kmat)] = t.flatten()
        amplitudes[i, np.isnan(kmat)] = np.nan + 1j * np.nan
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
