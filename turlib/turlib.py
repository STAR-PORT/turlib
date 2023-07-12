"""
Welcome to the TurLib - Turbulence Library for the estimation of integral turbulence parameters from AO telemetry.

Author: Nuno Morujão & Paulo Andrade

Feel free to use and expand this library to your own uses.
"""

import numpy as np
from scipy.optimize import leastsq
from turlib.fun_variance import nm, nz_variance, nz_covariance, agregate2alternate
import aotpy
import pandas as pd
import math
import warnings
from astropy.io import fits


def reader_aotpy(path: str, path_g_mat_sim, loop_instance=0, dimm_data: bool = False,
                 h_rad=4, l_rad=2, n_max_modes=15, diameter_tel=1.8,
                 conversion_modes_to_rad=1e-6 / (500 * 10 ** -9) * 2 * np.pi, estimate_noise=False, naomi=True):
    """
    returns vector for the application on turbulence estimation methods of turlib package
    for any aotpy system - defaults to NAOMI configuration

    Parameters
    ----------
    path:
        path to the system fits file - following aotpy translator conventions
        https://github.com/kYwzor/aotpy

    dimm_data:
        if enabled returns the atmospheric parameters
        of the dimm in a second vector

    loop_instance:
        grabs loop instance from aotpy system file, by default its 0 for NAOMI use case

    path_g_mat_sim:
        path to the generated simulated G matrix

    h_rad:
        Highest radial order in the fit

    l_rad:
        Lowest radial order in the fit

    n_max_modes:
        Maximum noll order included in the fit

    conversion_modes_to_rad:
        Conversion factor from pseudo mode units to radians (leave as 1 if already scaled)

    diameter_tel:
        Diameter of telescope in meters (use only if it isnt defined by aotpy)

    estimate_noise:
        Estimate noise through autocorrelation - if True will estimate

    naomi:
        Special flag for NAOMI integration with SPARTA convention of slope ordering

    Returns
    -------
    if dimm_data is False - returns turlib_vector
    if dimm_data is True - returns two vectors (turlib_vector, turbulence_parameters):

    turlib vector - input requirements for the estimator function (d, modes, ai2, noise_estimate, n_rec_modes, m, c_mat)
    turbulence parameters - seeing estimate of the dimm + telescope elevation

    """

    system = aotpy.AOSystem.read_from_file(path)  # AOSystem

    loop = system.loops[loop_instance]
    if not isinstance(loop, aotpy.ControlLoop):
        raise ValueError

    sensor: aotpy.WavefrontSensor = loop.input_sensor

    positions = loop.commands.data
    gradients = sensor.measurements.data

    dm2m_matrix: np.array = loop.commands_to_modes.data
    control_matrix: np.array = loop.measurements_to_modes.data

    # Simulated matrix import:

    gradient_matrix = fits.getdata(path_g_mat_sim)
    if naomi:
        # adopting SPARTA convention of alternated x and y slopes
        g_matrix_ordered = agregate2alternate(gradient_matrix)

        # rescaling matrix

        g_matrix_ordered[:, 0] = 0.0
        g_matrix_ordered = 0.5 / 0.375 * g_matrix_ordered

    if not naomi:
        g_matrix_ordered = gradient_matrix

    if loop.delay is None or loop.framerate is None:
        delay = 2
        warnings.warn('Either loop delay or framerate isnt correctly defined, assumed frame delay == 2')
    # adding frame delay to the shack-hartmann slopes (defaults to 2 if delay isn't present)
    else:
        delay = math.ceil(loop.framerate * loop.delay)

    data_size = len(positions)
    reformated_size = data_size - math.ceil(delay)
    interp_positions = positions[delay::1, :]
    slopes = gradients[0:-delay:1, :, :]
    j = n_max_modes

    residual_m = np.matmul(control_matrix[:, 0, :], slopes[:, 0, :].T) + np.matmul(control_matrix[:, 1, :],
                                                                                   slopes[:, 1, :].T)

    # Projection of commands on Slopes in arcsec

    dm_modes = np.matmul(dm2m_matrix, interp_positions.T)
    pseudo_open_modes_um = (residual_m + dm_modes)
    pseudo_open_modes = pseudo_open_modes_um * conversion_modes_to_rad

    if system.main_telescope.inscribed_diameter is not None:
        diameter = system.main_telescope.inscribed_diameter
    else:
        diameter = diameter_tel

    m = g_matrix_ordered.shape[1]
    bi = np.zeros([j, reformated_size])
    bi[1:] = pseudo_open_modes

    # reconstructed variances
    bi2 = np.var(bi, 1, ddof=1)  # ddof makes the denominator N-1, useful for the correct estimation of the variance.

    h_rad_ord = h_rad  # higher radial order in the fit
    l_rad_ord = l_rad  # lower radial order in the fit
    m_r0 = 0  # nbr of last radial orders excluded from the fit

    # modes to fit

    f_rad_ords = np.arange(l_rad_ord, h_rad_ord + 1 - m_r0)
    f_modes = np.array([], dtype=int)
    for iRadOrd in range(f_rad_ords.size):
        f_modes = np.append(f_modes, modes_of_radial_order(f_rad_ords[iRadOrd]))

    reconstructor_fitted = g_matrix_ordered[:, 0:j]
    inv_reconstructor_fitted = np.linalg.pinv(reconstructor_fitted)  # unrotated reconstructor
    reconstructor_remaining = g_matrix_ordered[:, j:m + 1]
    c = np.matmul(inv_reconstructor_fitted, reconstructor_remaining)  # cross-coupling matrix

    # estimate noise
    if estimate_noise:
        fusco_noise: np.array = noise_variance(bi)

    if not estimate_noise:
        fusco_noise: np.array = np.empty(len(bi2))

    if dimm_data:
        if system.atmosphere_params[0].fwhm[0] is not None and system.main_telescope.elevation is not None:
            return [diameter, f_modes, bi2, fusco_noise, j, m, c, reconstructor_fitted], \
                   [system.atmosphere_params[0].fwhm[0], system.main_telescope.elevation]
        else:
            return [diameter, f_modes, bi2, fusco_noise, j, m, c, reconstructor_fitted], [0, 0]
            warnings.warn('Either seeing or elevation werent defined - returns zeros array [0,0]')

    if not dimm_data:
        return [diameter, f_modes, bi2, fusco_noise, j, m, c, reconstructor_fitted]


def reader_modes(path: str, loop_instance=0, diameter_tel=1.8, n_max_modes=15,
                 conversion_modes_to_rad=1e-6 / (500 * 10 ** -9) * 2 * np.pi):
    """
    returns vector for the application on turbulence estimation methods of turlib package
    for any aotpy system - defaults to NAOMI configuration

    Parameters
    ----------
    path:
        path to the system fits file - following aotpy translator conventions
        https://github.com/kYwzor/aotpy

    loop_instance:
        grabs loop instance from aotpy system file, by default its 0 for NAOMI use case

    conversion_modes_to_rad:
        Conversion factor from pseudo mode units to radians (leave as 1 if already scaled)

    diameter_tel:
        Diameter of telescope in meters (use only if it isnt defined by aotpy)

    n_max_modes:
        Number of modes reconstructed from the telemetry

    Returns
    -------

    turlib vector - vector with:

         1) Diameter of telescope
         2) Modal variances
         3) Modal coefficients

    """

    system = aotpy.AOSystem.read_from_file(path)  # AOSystem

    loop = system.loops[loop_instance]
    if not isinstance(loop, aotpy.ControlLoop):
        raise ValueError

    sensor: aotpy.WavefrontSensor = loop.input_sensor

    positions = loop.commands.data
    gradients = sensor.measurements.data

    dm2m_matrix: np.array = loop.commands_to_modes.data
    control_matrix: np.array = loop.measurements_to_modes.data

    if loop.delay is None or loop.framerate is None:
        delay = 2
        warnings.warn('Either loop delay or framerate isnt correctly defined, assumed frame delay == 2')
    # adding frame delay to the shack-hartmann slopes (defaults to 2 if delay isn't present)
    else:
        delay = math.ceil(loop.framerate * loop.delay)

    data_size = len(positions)
    reformated_size = data_size - math.ceil(delay)
    interp_positions = positions[delay::1, :]
    slopes = gradients[0:-delay:1, :, :]
    j = n_max_modes
    residual_m = np.matmul(control_matrix[:, 0, :], slopes[:, 0, :].T) + np.matmul(control_matrix[:, 1, :],
                                                                                   slopes[:, 1, :].T)

    # Projection of commands on Slopes in arcsec

    dm_modes = np.matmul(dm2m_matrix, interp_positions.T)
    pseudo_open_modes_um = (residual_m + dm_modes)
    pseudo_open_modes = pseudo_open_modes_um * conversion_modes_to_rad

    if system.main_telescope.inscribed_diameter is not None:
        diameter = system.main_telescope.inscribed_diameter
    else:
        diameter = diameter_tel

    bi = np.zeros([j, reformated_size])
    bi[1:] = pseudo_open_modes

    # reconstructed variances
    bi2 = np.var(bi, 1, ddof=1)  # ddof makes the denominator N-1, useful for the correct estimation of the variance.

    return [diameter, bi2, bi]


def import_generated_matrix(path_g_mat_sim, wl, response=0.44, gradient_matrix_scale=0.22918311805232927,
                            header: bool = True):
    """
    Imports simulated gradient matrix for the import function

    Parameters
    ----------
    path_g_mat_sim: CSV matrix file
        Path to the stored matrix [slopes x modes]
    wl:
        Wavelength of the measurements - assumes 500 nm
    response:
        system response of the gradient matrix (leave as 1 if the matrix is already scaled)
    gradient_matrix_scale:
        Removes scale of the artificial matrix (leave as 1 if the matrix is correctly scaled)
    header: - Boolean
        If CSV file contains header == True (default)

    Returns
    -------
    Gradient matrix: np.array
        returns the transpose of the scaled simulated gradient matrix as an array

    """
    gradient_matrix = pd.read_csv(path_g_mat_sim)  # OOPAO Z2S matrix [24 X 200] -- NAOMI use case
    if header:
        gradient_matrix = np.asarray(gradient_matrix)[:, 1:]  # removing header
    if not header:
        gradient_matrix = np.asarray(gradient_matrix)
    gradient_matrix_rescaled = gradient_matrix * response / gradient_matrix_scale / wl

    return gradient_matrix_rescaled.T


def seeing_at_zenith(r0, meas_ang, wvl=500.0):
    """
    Parameters
    ----------
    r0:
        estimated r0 from the estimator functions
    meas_ang:
        altitude of the telescope at the time of the observation
    wvl:
        wavelength of the seeing estimate (defaults to 500nm)

    Returns
    -------
    seeing estimate at zenith
    """

    if isinstance(r0, np.ndarray):
        s = np.zeros(2)
        s[0] = 1.0 / r0[0]
        s[1] = 1.0 / r0[0] ** 2 * r0[1]
    else:
        s = 1.0 / r0

    seeing = (wvl / 500.0) ** (1. / 5.) * s * 0.9759 * wvl * 1e-9 * 180. / np.pi * 60 * 60

    return seeing * np.cos((90 - meas_ang) * np.pi / 180) ** (3 / 5)


def noise_variance(ai):
    """
    Parameters
    ----------
    ai:
        2d array with a sequence in time of Zernike coefficients from a set of modes.

    Returns
    -------
    Author: Paulo Andrade
    Zernike coefficient noise variance computation by the temporal autocorrelation method.
    Following the method described in Fusco 2004 (DOI: 10.1088/1464-4258/6/6/014)
    """

    n_modes, nps = ai.shape
    sp_fc = 1  # start point from center
    ep_fc = 4  # end point from center (center = nps)
    poly_order = 6
    si2_noise = np.zeros(n_modes)

    x = np.delete(np.arange(-ep_fc, ep_fc + 1), ep_fc)

    for iMode in range(n_modes):

        c_rec = np.correlate(ai[iMode, :], ai[iMode, :], "full") / nps
        c_points = np.concatenate((c_rec[nps - ep_fc - 1:nps - sp_fc], c_rec[nps + sp_fc - 1:nps + ep_fc]))
        c_fit = np.polyfit(x, c_points, poly_order)
        c_turb_0 = c_fit[poly_order]
        xx = c_rec[nps - 1] - c_turb_0

        if xx > 0:
            si2_noise[iMode] = xx

    return si2_noise


def cross_correction(n_rec_modes, m, c, ai_aj):
    """
    Parameters
    ----------
    n_rec_modes:
        number of modes in the reconstructor matrix (J)
    m:
        number of modes in the Zernike to slopes matrix
    c:
        Cross-coupling matrix iH*Hr (J x K) x (K x M - J) -> (J x (M - J))
    ai_aj:
        covariance matrix M x M

    Returns
    -------
    Author: Paulo Andrade
    cc     :
        variance correction 1st term
    cc_ct  :
        variance correction 2nd term (crossed term)

    """

    cc = np.zeros(n_rec_modes)
    cc_ct = np.zeros(n_rec_modes)

    for ii in range(2, n_rec_modes + 1):  # go through the modes (2,15) including 2 and 15.
        for jj in range(n_rec_modes + 1, m - n_rec_modes + 1):  # go through the non-corrected modes (J,M).

            jc = jj - n_rec_modes  # gives us a shifted version of the index (0, M - J).

            cc_ct[ii - 1] = cc_ct[ii - 1] + c[ii - 1, jc - 1] * ai_aj[ii - 1, jj - 1]

            '''
            Summing over the cross correlation, we set the cross correlation of the piston to 0 by default and as such 
            is not calculated in this code, this can be included by changing the way we go through our matrix 
            from 1 to 15 instead.
            '''

            for jl in range(n_rec_modes + 1, m - n_rec_modes + 1):
                # go through the non-corrected modes (J,M) for the non-crossed terms.
                jlc = jl - n_rec_modes
                cc[ii - 1] = cc[ii - 1] + c[ii - 1, jc - 1] * ai_aj[jj - 1, jl - 1] * c[ii - 1, jlc - 1]

    return cc + 2 * cc_ct


def modes_of_radial_order(n):
    """
    Parameters
    ----------
    n:
        radial order

    Returns
    -------
    Returns the array with the Noll modes of radial order n
    """

    return np.arange(n * (n + 1) / 2 + 1, (n + 1) * (n + 2) / 2 + 1, dtype=int)


def std_vector(h_rad_ord, l_rad_ord, fitted_var):
    """
    Parameters
    ----------
    h_rad_ord:
        Highest radial order included in fit
    l_rad_ord:
        Lowest radial order included in fit
    fitted_var:
        Fitted parameter - remaining and measurement noise removed

    Returns
    -------
    Author: Nuno Morujão
    standard deviation of the radial orders included in the fit.
    """

    std_v = np.zeros(h_rad_ord - l_rad_ord + 1)
    l_idx = 0  # last index
    for ii in range(len(std_v)):  # obtain the standard deviation of the radial orders
        f_idx = l_idx
        l_idx += len(modes_of_radial_order(l_rad_ord + ii))
        std_v[ii] = np.std(fitted_var[f_idx:l_idx])

    return std_v


def std_projection(h_rad_ord, l_rad_ord, standard_dev_vector):
    """
    Parameters
    ----------
    h_rad_ord:
        Highest radial order included in fit
    l_rad_ord:
        Lowest radial order included in fit
    standard_dev_vector:
        standard deviations of radial orders

    Returns
    -------
    Author: Nuno Morujão
    standard deviation vector for all azimuthal orders
    """

    std = np.array([])
    for ii in range(h_rad_ord - l_rad_ord + 1):
        size = len(modes_of_radial_order(l_rad_ord + ii))
        std = np.append(std, np.ones(size) * standard_dev_vector[ii])

    return std


"""functions imported from OOMAO"""


def n_modes_from_radial_order(n):
    """
    Parameters
    ----------
    n:
        zernike radial order = n

    Returns
    -------
    returns the number of Zernike polynomials (n+1)(n+2)/2 up to a given radial order n
    """

    return int((n + 1) * (n + 2) / 2)


def zernike_variance(d, p, x):
    """
    Function for estimator class:
    Calculates Zernike variances for the given turbulence parameters and telescope diameter

    Parameters
    ----------
    d:
        telescope diameter [m]
    p:
        turbulence parameters (r0, L0) output from estimator class
    x:
        noll modes to be estimated

    Returns
    -------
    Returns the theoretical Zernike variances for the set (r0, L0) and telescope diameter for the given modes
    """
    return nz_variance(p[0], p[1], d, x[-1])[[m - 1 for m in x]]


"""functions for iterative estimation of the turbulence parameters"""


def iterative_estimator(d, modes, ai2, noise_estimate, n_rec_modes, m, c_mat, n_iter=5, hro=4, lro=2,
                        full_vector=False):
    """
    Function that estimates the integrated turbulence parameters (r0, L0) from pseudo-open loop variances in a Zernike
    base.

    Parameters
    ----------
    d:
        diameter of telescope
    modes:
        modes [Noll convention] included in the fit
    ai2:
        pseudo-open loop variances
    noise_estimate:
        estimation of noise from noise_variance function (check documentation)
    n_rec_modes:
        Number of modes included in the reconstruction
    m:
        Total size of the c_mat matrix
    c_mat:
        Cross-talk matrix - specific to your system.
    n_iter:
        Number of iterations of the algorithm
    lro:
        Lowest radial order of fit
    hro:
        Highest radial order of fit
    full_vector:
        Full vector permits saving all turbulence parameter estimates

    Returns
    -------
    Authors: Nuno Morujão
    Returns the turbulence parameters (r0,l0) fitted from the turbulence estimator class
    If full vector True:
        returns r0,L0 for every iteration and final fitted variances

    If full vector False:
        returns r0,L0 for final iteration and final fitted variances
    """

    # initial estimation without cross talk correction
    tp = TurbulenceEstimator(d, modes, ai2, si2_nn=noise_estimate)

    r0 = tp.tp[0]
    l0 = tp.tp[1]

    if full_vector:
        r0_vector = np.zeros(n_iter + 1)
        l0_vector = np.zeros(n_iter + 1)

        r0_vector[0] = r0
        l0_vector[0] = l0

    for vv in range(n_iter):
        # Calculation of the remaining error contributions

        aiaj_0 = nz_covariance(r0, l0, d, m)

        si2_cc_1 = cross_correction(n_rec_modes, m + n_rec_modes, c_mat, aiaj_0)

        tp = TurbulenceEstimator(d, modes, ai2, si2_nn=noise_estimate, si2_cc=si2_cc_1,
                                 h_rad_ord=hro, l_rad_ord=lro)

        r0 = tp.tp[0]
        l0 = tp.tp[1]

        if full_vector:
            r0_vector[vv + 1] = r0
            l0_vector[vv + 1] = l0

    if full_vector:
        return r0_vector, l0_vector, tp.fitted_ai2

    if not full_vector:
        return r0, l0, tp.fitted_ai2


def full_uncertainty_estimator(d, modes, ai2, noise_estimate, n_rec_modes, m, c_mat, l_rad_ord=2, h_rad_ord=4,
                               n_samples=50, n_iter=5):
    """
    Function that estimates the integrated turbulence parameters (r0, L0) from pseudo-open loop variances in a Zernike
    base. It additionally gives uncertainty estimates for the estimated (r0, L0)

    Parameters
    ----------
    d:
        diameter of telescope
    modes:
        modes [Noll convention] included in the fit
    ai2:
        pseudo-open loop variances
    noise_estimate:
        estimation of noise from noise_variance function (check documentation)
    n_rec_modes:
        Number of modes included in the reconstruction
    m:
        Total size of the c_mat matrix
    c_mat:
        Cross-talk matrix - specific to your system.
    n_iter:
        Number of iterations of the algorithm
    l_rad_ord:
        lowest radial order fit
    h_rad_ord:
        highest radial order fit
    n_samples:
        number of samples used to calculate the uncertainty (50 by default)

    Returns
    -------
    Authors: Nuno Morujão
    Estimated turbulence parameters with an added uncertainty estimate (r0, u(r0)) (L0, u(L0)) in units of [m]
    """

    r0_vector = np.zeros(n_samples)
    l0_vector = np.zeros(n_samples)
    s_idx = n_modes_from_radial_order(l_rad_ord - 1)
    standard_deviations = std_projection(h_rad_ord, l_rad_ord, std_vector(h_rad_ord, l_rad_ord, ai2[3:]))
    r0_i, l0_i = iterative_estimator(d, modes, ai2, noise_estimate, n_rec_modes, m, c_mat, n_iter=n_iter)[:2]
    modal_vector = np.zeros(n_rec_modes)

    for kk in range(n_samples):
        modal_vector[s_idx:] = np.random.normal(ai2[s_idx:], standard_deviations)
        r0, l0 = iterative_estimator(d, modes, modal_vector, noise_estimate, n_rec_modes, m, c_mat, n_iter=n_iter)[:2]

        r0_vector[kk] = r0
        l0_vector[kk] = l0

    # Extra sqrt factor so to conform to an error around a mean
    error_r0 = np.nanstd(r0_vector) / np.sqrt(n_samples)
    error_l0 = np.nanstd(l0_vector) / np.sqrt(n_samples)

    return [r0_i, error_r0], [l0_i, error_l0], r0_vector


class TurbulenceEstimator:
    """
    Turbulence_Estimator: Turbulence Parameters estimation - class to estimate
    r0 and L0. The Zernike coefficient (ZC) variances are fitted by the theoretical
    von Karman ZC variances.

    Parameters
    ----------
    d:
        telescope diameter
    modes:
        vector with Noll modes to use in the fit
    modes_excluded:
        Particular Noll modes to be excluded from the fit
    ai2:
        vector with ZC variances
    si2_nn:
        vector with ZC noise variances
    si2_cc:
        vector with cross-coupling corrections to ZC variances
    l_rad_ord:
        the lowest radial order included in the fit
    h_rad_ord:
        the highest radial order included in the fit

    Returns
    -------
    fit:
        scipy.optimize.leastsq output
    tp:
        vector with parameter estimates
    fitted_ai2:
        fitted variances
    fitted_si2_nn:
        noise variances of fitted modes
    fitted_si2_cc:
        cross-coupling corrections for the fitted modes
    fitted_nn:
        fitted modes radial order
    fitted_mm:
        fitted modes azimuthal order
    std_v:
        standard deviations within radial orders
    std:
        projected standard deviation to all Noll modes of the order
    """

    def __init__(self, d, modes, ai2, si2_nn, si2_cc=None, h_rad_ord=4, l_rad_ord=2, modes_excluded=np.array([])):

        self.D = d
        self.h_rad_ord = h_rad_ord
        self.l_rad_ord = l_rad_ord
        self.modes = modes  # (numbering assumes index 1 as piston)
        self.modes_excluded = modes_excluded
        for ime in range(modes_excluded.size):
            self.modes = self.modes[self.modes != self.modes_excluded[ime]]

        self.fitted_ai2 = ai2[[m - 1 for m in self.modes]]
        self.fitted_si2_cc = si2_cc
        self.fitted_si2_nn = si2_nn
        self.fitted_nn = [nm(m)[0] for m in self.modes]
        self.fitted_mm = [nm(m)[1] for m in self.modes]

        # removes contribution of estimated remaining noise
        if self.fitted_si2_cc is not None:
            self.fitted_si2_cc = si2_cc[[m - 1 for m in self.modes]]
            self.fitted_ai2 = self.fitted_ai2 - self.fitted_si2_cc

        # removes contribution of estimated measurement noise
        if self.fitted_si2_nn is not None:
            self.fitted_si2_nn = si2_nn[[m - 1 for m in self.modes]]
            self.fitted_ai2 = self.fitted_ai2 - self.fitted_si2_nn

        '''

        From here we remove the first version of the of the noise estimate.

        '''

        # Parameters initial guess - we randomize the positions in order not to induce bias in the fitting
        # Recommended small initial parameters - from the chi squared map of the algorithm for an on-sky sample

        self.p0 = np.array([.01 + np.random.rand() * .01, np.random.random() * 4 + 1])

        # weight vector for the chi square

        # Obtain the standard deviations within radial orders
        self.std_v = std_vector(self.h_rad_ord, self.l_rad_ord, self.fitted_ai2)

        # Project the standard deviation to all Noll modes of the order
        self.std = std_projection(self.h_rad_ord, self.l_rad_ord, self.std_v)

        # fit function - theoretical zernike Variances as a function of turbulence parameters, r0 and L0
        def af1(x, p):
            return zernike_variance(self.D, p, x)

        '''

        Here we define the fitting curve to be weighed by the standard deviation of the radial modes
        Serves as a more natural way of performing the least squares algorithm, avoiding the logarithm approach

        We need to recalculate the deviation every time we calculate the new fitted_ai2. As the points shift 
        in place.

        '''

        def af2(p, x, y):
            return (af1(x, p) - y) / self.std

        '''
        af2 uses 2 things;

        calculation of variances from current estimate of r0 and L0 and removes noise from it - Artificial

        subtracts the current fitted variances - Real

        Results in a residual of the model vs real data.
        '''

        self.fit = leastsq(af2, self.p0, args=(self.modes, self.fitted_ai2), full_output=True)

        self.tp = self.fit[0]
