import numpy as np
from scipy.special import gamma


def zern_index(j):
    """
    Find the [n,m] list giving the radial order n and azimuthal order
    of the Zernike polynomial of Noll index j.

    Parameters
    ----------
    j:
        Noll index

    Returns
    -------
    n:
        Noll radial order
    m:
        Noll azimuthal order
    """
    n = int((-1. + np.sqrt(8 * (j - 1) + 1)) / 2.)
    p = (j - (n * (n + 1)) / 2.)
    k = n % 2
    m = int((p + k) / 2.) * 2 - k

    if m != 0:
        if j % 2 == 0:
            s = 1
        else:
            s = -1
        m *= s

    return [n, m]


def nm(j, sign=0):
    """
    returns the [n,m] list giving the radial order n and azimutal order of the zernike polynomial of index j
    if sign is set, will also return a 1 for cos, -1 for sine or 0 when m==0.

    Parameters
    ----------
    j:
        Noll index
    sign:
        determines sign of cosine (1,-1,0)

    Returns
    -------
    n:
        Noll radial order
    m:
        Noll azimuthal order
    """

    n = int((-1. + np.sqrt(8 * (j - 1) + 1)) / 2.)
    p = (j - (n * (n + 1)) / 2.)
    k = n % 2
    m = int((p + k) / 2.) * 2 - k
    if sign == 0:
        return [n, m]
    else:  # determine whether is sine or cos term.
        if m != 0:
            if j % 2 == 0:
                s = 1
            else:
                s = -1
            # nn,mm=nm(j-1)
            # if nn==n and mm==m:
            #    s=-1
            # else:
            #    s=1
        else:
            s = 0
        return [n, m, s]


"""
The following functions are adapted from the OOMAO package 
Developed by R. Conan and C. Correia, 
Object-oriented Matlab adaptive optics toolbox (https://ui.adsabs.harvard.edu/abs/2014SPIE.9148E..6CC/abstract)

Results have been validated against OOMAO outputs.
"""


def agregate2alternate(G):
    """
    inverse operation of alternate2agregate
    """
    a0l, a1l = G.shape
    Ga = np.zeros((a0l, a1l))

    nS = int(a0l / 2)

    for ii in range(nS):
        Ga[2 * ii, :] = G[ii, :]
        Ga[2 * ii + 1, :] = G[ii + nS, :]

    return Ga


def new_gamma(a, b):
    """
    Function imported from OOMAO
    R. Conan and C. Correia, Object-oriented Matlab adaptive optics toolbox,
    roceedings of the SPIE, Volume 9148, id. 91486C 17 pp. (2014).
    https://ui.adsabs.harvard.edu/abs/2014SPIE.9148E..6CC/abstract
    """
    return np.prod(gamma(a)) / np.prod(gamma(b))


def un_param_ex4q2(mu, alpha, beta, p, a):
    """
    Function imported from OOMAO
    R. Conan and C. Correia, Object-oriented Matlab adaptive optics toolbox,
    roceedings of the SPIE, Volume 9148, id. 91486C 17 pp. (2014).
    https://ui.adsabs.harvard.edu/abs/2014SPIE.9148E..6CC/abstract


    Parameters
    ----------

    Returns
    -------
    ret:
        Computes the integral given by the Eq.(2.33) of the thesis of R. Conan (Modelisation des effets de l'echelle externe de coherencespatiale du front d'onde pour l'Observation a Haute Resolution Angulaire en Astronomie, University of Nice-Sophia Antipolis, October 2000) http://www-astro.unice.fr/GSM/Bibliography.html#thesis
    """
    a1 = [(alpha + beta + 1) / 2, (2 + mu + alpha + beta) / 2, (mu + alpha + beta) / 2]
    b1 = [1 + alpha + beta, 1 + alpha, 1 + beta]
    a2 = [(1 - mu) / 2 + p, 1 + p, p]
    b2 = [1 + (alpha + beta - mu) / 2 + p, 1 + (alpha - beta - mu) / 2 + p, 1 + (beta - alpha - mu) / 2 + p]

    return (1 / (2 * np.sqrt(np.pi) * gamma(p))) * (
            new_gamma(np.append(a1, p - (mu + alpha + beta) / 2), b1) * a ** (mu + alpha + beta) *
            pochammer_series(3, 5, a1, [1 - p + (mu + alpha + beta) / 2, 1 + alpha + beta, 1 + alpha, 1 + beta, 1],
                             a ** 2) +
            new_gamma(np.append((mu + alpha + beta) / 2 - p, a2), b2) * a ** (2 * p) *
            pochammer_series(3, 5, a2, [1 - (mu + alpha + beta) / 2 + p, 1 + (alpha + beta - mu) / 2 + p,
                                        1 + (alpha - beta - mu) / 2 + p, 1 + (beta - alpha - mu) / 2 + p, 1],
                             a ** 2))


def pochammer_series(p, q, a, b, z, tol=1e-6, nmax=1e3):
    """
    Function imported from OOMAO
    R. Conan and C. Correia, Object-oriented Matlab adaptive optics toolbox,
    roceedings of the SPIE, Volume 9148, id. 91486C 17 pp. (2014).
    https://ui.adsabs.harvard.edu/abs/2014SPIE.9148E..6CC/abstract

    Parameters
    ----------

    Returns
    -------
    ret:
        Computes the Pochammer series as defined in the thesis of R. Conan (Modelisation des effets de l'echelle externe de coherencespatiale du front d'onde pour l'Observation a Haute Resolution Angulaire en Astronomie, University of Nice-Sophia Antipolis, October 2000) http://www-astro.unice.fr/GSM/Bibliography.html#thesis
    """

    if (p == (q + 1) and np.abs(z) < 1) or (np.abs(z) == 1 and np.real(np.sum(a) - np.sum(b)) < 0) or p < (q + 1):
        # print('len(a) =',len(a))
        # print('len(b) =',len(b))
        # print('p =',p)
        # print('q =',q)
        if p == len(a) and q == len(b):

            if np.size(z) != 1:
                raise Exception('z should be a number -> pi*alpha*D/L0')
            out = 0  # np.zeros(np.size(z)) - z is a number, why was this approach taken here ?

            if z == 0:
                out = 1

            if z != 0:
                ck = 1
                step = np.infty
                k = 0
                som = ck
                while (k <= nmax) and (step > tol):
                    ckp1 = np.prod([x + k for x in a]) * z * ck / np.prod([x + k for x in b])
                    step = abs(abs(ck) - abs(ckp1))
                    som = som + ckp1
                    k = k + 1
                    ck = ckp1
                if step > tol:
                    print('pochammerSeries', 'Maximum iteration reached before convergence')

                out = som

        else:
            raise Exception('p and q must be the same length than vectors a and b, respectively')

    else:
        raise Exception('This generalized hypergeometric function doesn''t converge')
    return out


def nz_cov_coeff(r0, l0, diam, i, n_rec_modes):
    """
    Function imported from OOMAO
    R. Conan and C. Correia, Object-oriented Matlab adaptive optics toolbox,
    roceedings of the SPIE, Volume 9148, id. 91486C 17 pp. (2014).
    https://ui.adsabs.harvard.edu/abs/2014SPIE.9148E..6CC/abstract

    Parameters
    ----------
    r0:
        Fried parameter
    l0:
        outer scale of turbulence
    diam:
        diameter of telescope
    i:
        Zernike coefficient index
    n_rec_modes:
        number of reconstructed modes (j)

    Returns
    -------
    computes the covariance matrix of Zernike coefficients from the Zernike polynomials
    """
    # For when we don't have a zernike object. (Estimating parameters will use this function)

    from scipy.special import gamma
    ni, mi = zern_index(i)
    nj, mj = zern_index(n_rec_modes)
    cov = 0

    if (mi == mj) and (abs(i - n_rec_modes) % 2 == 0 or ((mi == 0) and (mj == 0))):
        if l0 == np.infty:

            if i == 1 and n_rec_modes == 1:
                cov = np.infty  # piston is infinite with an infinite outer scale (we can't force it by equation)

            else:

                cov = (gamma(11 / 6) ** 2 * gamma(14 / 3) / (2 ** (8 / 3) * np.pi)) * (24 * gamma(6 / 5) / 5) ** (
                        5. / 6) * \
                      (diam / r0) ** (5. / 3) * np.sqrt((ni + 1) * (nj + 1)) * (-1) ** ((ni + nj - mi - mj) / 2) * \
                      new_gamma(-5 / 6 + (ni + nj) / 2,
                                [23 / 6 + (ni + nj) / 2, 17 / 6 + (ni - nj) / 2, 17 / 6 + (nj - ni) / 2])

        if l0 != np.infty:  # Here we assume that there is only a single atmospheric layer.

            alpha = 1  # - atm.altitude[kLayer]/src.height # we take the source to be in the infinity

            cov += (4 * gamma(11. / 6) ** 2 / np.pi ** (14. / 3)) * (24 * gamma(6 / 5) / 5) ** (5 / 6) * \
                   (l0 / r0) ** (5 / 3) * (l0 / (alpha * diam)) ** 2 * \
                   np.sqrt((ni + 1) * (nj + 1)) * (-1) ** ((ni + nj - mi - mj) / 2) * \
                   un_param_ex4q2(0, ni + 1, nj + 1, 11 / 6, np.pi * alpha * diam / l0)

    else:
        cov = 0

    return cov


def nz_covariance(r0, l0, diam, n_rec_modes):
    """

    Parameters
    ----------
    r0:
        Fried parameter
    l0:
        outer scale
    diam:
        diameter of telescope
    n_rec_modes:
        number of reconstructed modes (j)

    Returns
    -------
    Computes the covariance matrix from the coefficients calculated in nz_cov_coeff function
    """

    cov_value = np.zeros([n_rec_modes, n_rec_modes])

    '''
    For now we will use an element to element operation on the covariance operation, there is a bit of a problem in the
    calculus utilizing vectors for the covcoef functions, due to the inablity of the ao.zern.zernindex to handle vectors

    '''

    for ii in range(n_rec_modes):
        for jj in range(n_rec_modes):
            cov_value[ii][jj] = nz_cov_coeff(r0, l0, diam, ii + 1, jj + 1)

    return cov_value


def nz_variance(r0, l0, diam, n_rec_modes):
    """

    Parameters
    ----------
    r0:
        Fried parameter
    l0:
        outer scale
    diam:
        diameter of telescope
    n_rec_modes:
        number of reconstructed modes (j)

    Returns
    -------
    Computes the variance vector from the coefficients calculated in nz_cov_coeff function
    """

    var = np.zeros(n_rec_modes)

    for ii in range(n_rec_modes):
        jj = ii + 1  # included piston in calculus, matlab typically does not include

        var[ii] = nz_cov_coeff(r0, l0, diam, jj, jj)

    return var
