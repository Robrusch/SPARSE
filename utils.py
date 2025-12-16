# -*- coding: utf-8 -*-
"""
Created on Wed May 28 15:11:17 2025

@author: Roberto Bruschini
"""
import numpy as np
import pandas as pd
import warnings
from scipy.linalg import inv
from scipy.interpolate import AAA


def poles(k_matrix):
    """
    Calculates scattering poles from the K-matrix.
    Note that this function returns the nominal masses and widths of the poles, not the physical resonance parameters.
    
    Parameters
    ----------
    k_matrix : DataFrame
        Pandas DataFrame containing flattened K-matrices for various energies.
        This is typically an output of the function k_matrices from the SPARSE module.
    
    Returns
    -------
    DataFrame
        Pandas DataFrame containing the nominal masses, widths, and couplings of the scattering poles.
    
    """
    kmat = k_matrix.dropna(axis=1, how='all')
    assert not kmat.isna().any(axis=None), 'Input values span across one or multiple thresholds. Try excluding threshold values by using DataFrame.loc[Emin:Emax].'
    x = kmat.index.to_numpy()
    n = int(np.sqrt(len(kmat.columns)))
    y = kmat.to_numpy().reshape(-1, n, n)
    sign_change = np.all(y[:-1] * y[1:] < 0, axis=(1,2))
    large = np.all(np.abs(y[:-1]) > 1, axis=(1,2))
    n_poles = np.count_nonzero(sign_change & large)
    assert n_poles > 0, 'Poles could not be detected. Check your input K-matrix.'
    poles = np.empty((n_poles, n, n))
    residues = np.empty_like(poles)
    for i in range(n):
        for j in range(n):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                r = AAA(x, y[:, i, j], max_terms=n_poles + 1)
            assert np.isreal(r.poles).all() and np.isreal(r.residues).all(), 'Polynomial interpolation yields complex poles or residues. Check your input K-matrix.'
            order = np.argsort(r.poles().real)
            poles[:, i, j] = r.poles().real[order]
            residues[:, i, j] = r.residues().real[order]
    masses = poles[:, 0, 0]
    assert np.allclose(poles, masses[:, np.newaxis, np.newaxis]), 'Incompatible pole positions between different channels. Check your input K-matrix.'
    res_diag = np.diagonal(residues, axis1=1, axis2=2)
    assert np.all(res_diag <= 0), f'Non-resonant pole(s) at {masses[np.any(res_diag > 0, axis=1)]} detected. Try excluding non-resonant pole(s) by using DataFrame.loc[Emin:Emax].'
    res_trace = np.sum(res_diag, axis=1)
    widths = -2 * res_trace
    print(res_trace.shape)
    couplings = np.sqrt(res_diag / res_trace[:, np.newaxis])
    couplings[:, 1:] *= -np.sign(residues[:, 1:, 0])
    if not np.allclose(couplings[:,np.newaxis] * couplings[..., np.newaxis], residues / res_trace[:, np.newaxis, np.newaxis]):
        warnings.warn('Factorization theorem might not satisfied.', stacklevel=2)
    decay_channels = kmat.columns.remove_unused_levels().levels[0]
    data = np.hstack([masses[:, np.newaxis], widths[:, np.newaxis], couplings])
    labels = ['Mass', 'Width'] + [f'Coupling {i}' for i in decay_channels]
    results = pd.DataFrame(data=data, columns=labels)
    return results


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
