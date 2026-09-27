'''
    Author: Axel Raboonik
    Email : raboonik@gmail.com
    
    Github: https://github.com/raboonik
    
    Article: https://iopscience.iop.org/article/10.3847/1538-4357/adc917
    
    Description: The characteristic speeds and the eigenvector quantities of Paper III (beta, s_q, alpha_s,
                 alpha_f), with the conventions at the degenerate points of the eigensystem
'''

import numpy as np

def slow_fast_speeds(aq2, ap2, c2):
    '''
        Fast and slow magnetosonic speeds along one axis, from the Alfven speed squared along the
        axis (aq2), perpendicular to it (ap2), and the sound speed squared (c2). Written without
        subtracting nearly equal numbers, so both stay accurate (and never negative) for weak or
        strong fields, fields nearly along the axis, and a ~ c:
            D    = (aq2 + ap2 + c2)^2 - 4 aq2 c2 = (aq2 - c2)^2 + ap2 (ap2 + 2 aq2 + 2 c2) >= 0
            cf^2 = (aq2 + ap2 + c2 + sqrt(D)) / 2
            cs^2 = aq2 c2 / cf^2                  (since cf^2 cs^2 = aq2 c2)
    '''
    cf2 = 0.5 * (aq2 + ap2 + c2 + np.sqrt((aq2 - c2)**2 + ap2 * (ap2 + 2*aq2 + 2*c2)))
    cs2 = np.divide(aq2 * c2, cf2, out=np.zeros_like(cf2), where=cf2 > 0)
    return np.sqrt(cf2), np.sqrt(cs2)

# Conventions at the degenerate points of the Roe & Balsara (1996) eigensystem used in Paper III.
# They only decide how energy is split between modes that cannot be told apart there; the totals
# (Alfven + slow + fast, and the sum over all modes) do not depend on them.

def perp_basis(a1, a2):
    '''
        a_perp = |(a1, a2)| and the unit vector (beta_1, beta_2) = (a1, a2) / a_perp perpendicular to
        the axis, with beta_1 = beta_2 = 1/sqrt(2) where a_perp = 0 (Paper III, Eq. 7).
    '''
    ap    = np.hypot(a1, a2)                          # no under/overflow, unlike sqrt(a1**2 + a2**2)
    beta1 = np.divide(a1, ap, out=np.full_like(ap, 2**-0.5), where=ap > 0)
    beta2 = np.divide(a2, ap, out=np.full_like(ap, 2**-0.5), where=ap > 0)
    return ap, beta1, beta2

def sgn(a):
    '''sgn(a) = +1 or -1, never 0: s_q = 0 would make the Alfven (and slow) eigenvector pairs linearly dependent.'''
    return np.copysign(1.0, a)

def slow_fast_alphas(aq2, ap2, c2):
    '''
        alpha_s, alpha_f (Paper III, after Eqs. 6) without subtracting nearly equal numbers. With
        d = a^2 - c^2 and sqrt(D) = sqrt(d^2 + 4 ap2 c2) = cf^2 - cs^2:
            alpha_s^2 = (sqrt(D) + d) / (2 sqrt(D)),   alpha_f^2 = (sqrt(D) - d) / (2 sqrt(D)),
        where whichever of the two would cancel is written as 2 ap2 c2 / (sqrt(D) (sqrt(D) + |d|)).
        At the triple point (ap2 = 0 and aq2 = c2, so D = 0) both are set to 1/sqrt(2).
    '''
    d     = aq2 + ap2 - c2
    sqD   = np.sqrt(d**2 + 4*ap2*c2)
    ad    = np.abs(d)
    big   = np.divide(sqD + ad, 2*sqD, out=np.full_like(sqD, 0.5), where=sqD > 0)
    small = np.divide(2*ap2*c2, sqD*(sqD + ad), out=np.full_like(sqD, 0.5), where=sqD > 0)
    alpha_s = np.sqrt(np.where(d >= 0, big, small))
    alpha_f = np.sqrt(np.where(d >= 0, small, big))
    return alpha_s, alpha_f
