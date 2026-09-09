# -*- coding: utf-8 -*-
"""
Created on Wed May 28 15:11:17 2025

@author: Roberto Bruschini
"""
import numpy as np
import pandas as pd
import warnings
from scipy.integrate import simpson
from scipy.linalg import inv
from scipy.interpolate import AAA


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
        t = k @ inv(np.eye(n) - 1j * k)
        amplitudes[i, ~np.isnan(kmat)] = t.flatten()   
    return pd.DataFrame(amplitudes,
                        index=k_matrix_df.index,
                        columns=k_matrix_df.columns)


def poles(amplitudes, rtol=1e-4, in_interval=True, extremes_exclusion=5e-2, pole_spread_tol=1e-3):
    """
    Calculates complex poles from the scattering amplitudes for real energies.
    This function extrapolates the scattering amplitude to complex energies using
    the AAA algorithm for rational polynomial interpolation.
    
    Parameters
    ----------
    amplitudes : DataFrame
        Pandas DataFrame containing flattened scattering amplitudes for various energies.
    rtol: float, optional
        Relative tolerance in the AAA algorithm. See the documentation of
        scipy.optimize.AAA for further information. Default is 1e-4.
    in_interval: bool, optional
        Wether to discard extrapolated poles whose real part is found outside
        of the input energy region. Default is True.
    extremes_exclusion: float, optional
        If in_interval is set to True, further exclude extrapolated poles whose
        real part is found to close to the extremes of the input energy region.
        Default is do create an exclusion region next to each extreme equal to
        5% of the total length of the energy interval.
    pole_spread_tol: float, optional
        Relative tolerance for the spread of the extrapolated poles found in
        different channels. If the tolerance is exceeded, a warning is made and
        the raw data of the pole positions is printed out. Default is 1e-3.  

    Returns
    -------
    DataFrame
        Pandas DataFrame containing the positions and residues of the scattering poles.
    
    """
    amps = amplitudes.dropna(axis=1, how='all')
    assert not amps.isna().any(axis=None), 'Input values span across one or multiple thresholds. Try excluding threshold values by using DataFrame.loc[Emin:Emax].'
    x = amps.index.to_numpy()
    n = int(np.sqrt(len(amps.columns)))
    y = amps.to_numpy().reshape(-1, n, n)
    if in_interval:
        xmin = x[0]
        xmax = x[-1]
        if extremes_exclusion > 0:
            exclude = (xmax - xmin) * extremes_exclusion
            xmin += exclude
            xmax -= exclude
    else:
        xmin = -np.inf
        xmax = np.inf
    poles_matrix = []
    residues = []
    for i in range(n):
        pole_row = []
        residue_row = []
        for j in range(n):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                r = AAA(x, y[:, i, j], rtol=rtol)
            poles_inside = np.logical_and(r.poles().real > xmin,
                r.poles().real < xmax)
            new_poles = r.poles()[poles_inside]
            new_residues = r.residues()[poles_inside]
            order = np.argsort(new_poles.real)
            pole_row.append(new_poles[order])
            residue_row.append(new_residues[order])
        poles_matrix.append(pole_row)
        residues.append(residue_row)
    try:
        poles_matrix = np.array(poles_matrix).transpose((2, 0, 1))
        residues = np.array(residues).transpose((2, 0, 1))
    except ValueError:
        raise RuntimeError("Found inconsistent number of poles in different channels.")
    poles_diag = np.diagonal(poles_matrix, axis1=1, axis2=2)
    res_diag = np.diagonal(residues, axis1=1, axis2=2)
    poles = np.average(poles_diag, axis=1, weights=np.abs(res_diag))
    if not np.allclose(np.expand_dims(poles, axis=(1,2)), poles_matrix, rtol=pole_spread_tol):
        warnings.warn('Pole position spread across channels exceeds the input tolerance. Check the pole positions below.', stacklevel=2)
        print(poles_matrix)
    cols = amplitudes.columns.remove_unused_levels().rename(['Residue row', 'Residue column'])
    results = pd.DataFrame(data=residues.reshape(-1, n**2),
        index=pd.Index(poles,name='Pole position'),
        columns=cols)
    return results


def shifts(k_matrix_df):
    """
    Calculate the scattering phase shifts and inelasticities for different
    energies from the input K-matrix values.
    
    Parameters
    ----------
    k_matrix_df : DataFrame
        Pandas DataFrame containing flattened K-matrices for various energies.
        This is typically an output of the function k_matrices from the SPARSE
        module.

    Returns
    -------
    DataFrame, DataFrame
        Two Pandas DataFrames containing the flattened scattering phase shifts
        and inelasticities, respectively. Values corresponding to closed
        channels are set to NaN.
    
    """
    n_max = int(np.sqrt(k_matrix_df.shape[1]))
    labels = k_matrix_df.columns.remove_unused_levels().levels[0].rename('')
    shifts = np.full((len(k_matrix_df), n_max), np.nan)
    modulo_pi = np.zeros_like(shifts)
    inelasticities = np.full(k_matrix_df.shape, np.nan)
    sign_switch = np.ones_like(inelasticities)
    for i, kmat in enumerate(k_matrix_df.to_numpy()):
        kflat = kmat[~np.isnan(kmat)]
        n = int(np.sqrt(len(kflat)))
        k = kflat.reshape(n, n)
        s = (np.eye(n) + 1j * k) @ inv(np.eye(n) - 1j * k)
        shifts[i, :n] = np.angle(np.diagonal(s)) / 2
        inelasticities[i, ~np.isnan(kmat)] = abs(s).flatten() 
        if i > 0:
            skip = shifts[i] - shifts[i-1]
            skip_up = skip > 1
            skip_down = skip < -1
            modulo_pi[i:, skip_up] -= np.pi
            modulo_pi[i:, skip_down] += np.pi
        if i > 2:
            small = inelasticities[i-1] < 1e-3
            v_shaped = np.logical_and(inelasticities[i-1] < inelasticities[i-2],
                                      inelasticities[i-1] < inelasticities[i])
            der_skip = np.logical_and(small, v_shaped)
            sign_switch[i:, der_skip] *= -1
    shifts_df = pd.DataFrame(shifts + modulo_pi,
                             index=k_matrix_df.index,
                             columns=labels)
    inelasticities_df = pd.DataFrame(inelasticities * sign_switch,
                                     index=k_matrix_df.index,
                                     columns=k_matrix_df.columns)
    for i, label1 in enumerate(k_matrix_df.columns.remove_unused_levels().levels[0]):
        for j, label2 in enumerate(k_matrix_df.columns.remove_unused_levels().levels[1]):
            if j <= i:
                del inelasticities_df[label1, label2]
    return shifts_df, inelasticities_df


def composition(wavefunc_df, sort=False):
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
    prob = simpson(psi_squared, wavefunc_df.index.to_numpy(), axis=0)
    if sort:
        order = np.argsort(prob)[::-1]
        data = prob[order]
        columns = wavefunc_df.columns[order]
    else:
        data = prob
        columns = wavefunc_df.columns
    return pd.Series(data, columns)
