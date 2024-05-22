import numpy as np
import torch


# Universal constants
C = 340. * 100.  # Speed of sound in air (cm/s)


def compute_length_of_air_column_cylindrical(
        timestamps, duration, height, b, **kwargs,
    ):
    """
    Randomly chooses a l(t) curve satisfying the two point equations.
    """
    L = height * ( (1 - np.exp(b * (duration - timestamps))) / (1 - np.exp(b * duration)) )
    return L


def compute_axial_frequency_cylindrical(
        lengths, radius, beta=0.62, mode=1, **kwargs,
    ):
    """
    Computes axial resonance frequency for cylindrical container at given timestamps.
    """
    if mode == 1:
        harmonic_weight = 1.
    elif mode == 2:
        harmonic_weight = 3.
    elif mode == 3:
        harmonic_weight = 5.
    else:
        raise ValueError

    # Compute fundamental frequency curve
    F0 = harmonic_weight * (0.25 * C) * (1. / (lengths + (beta * radius)))

    return F0


def compute_axial_frequency_bottleneck(
        lengths, radius, height, Rn, Hn, beta_bottle=(0.6 + 8/np.pi), **kwargs,
    ):
    # Here, R and H are base radius and height of the bottleneck
    eps = 1e-6
    kappa = (0.5 * C / np.pi) * (Rn/radius) * np.sqrt(1 / (Hn + beta_bottle * Rn))
    frequencies = kappa * np.sqrt(1 / (lengths + eps))
    return frequencies


def compute_f0_cylindrical(Y, rho_g, a, R, H, mode=1, **kwargs,):

    if mode == 1:
        m = 1.875
        n = 2
    elif mode == 2:
        m = 4.694
        n = 3
    elif mode == 3:
        m = 7.855
        n = 4
    else:
        raise ValueError

    term = ( ((n**2 - 1)**2) + ((m * R/H)**4) ) / (1 + (1./n**2))
    f0 = (1. / (12 * np.pi)) * np.sqrt(3 * Y / rho_g) * (a / (R**2)) * np.sqrt(term)
    return f0


def compute_xi_cylindrical(rho_l, rho_g, R, a, **kwargs,):
    """
    Different papers use different multipliers.
    For us, using 12. * (4./9.) works best empirically.
    """
    xi = 12. * (4. / 9.) * (rho_l/rho_g) * (R/a)
    return xi


def compute_radial_frequency_cylindrical(
        heights, R, H, Y, rho_g, a, rho_l, power=3, mode=1, **kwargs,
    ):
    """
    Computes radial resonance frequency for cylindrical.

    Args:
        heights (np.ndarray): height of liquid at pre-defined time stamps
    """
    # Only f0 changes for higher modes
    f0 = compute_f0_cylindrical(Y, rho_g, a, R, H, mode=mode)
    xi = compute_xi_cylindrical(rho_l, rho_g, R, a)
    frequencies = f0 / np.sqrt(1 + xi * ((heights/H) ** power) )
    return frequencies


def get_frequencies(
        t, params, container_shape="cylindrical", harmonic=None, vibration_type="axial",
    ):
    """
    Computes requires frequency f(t) for given t.
    """
    
    if container_shape == "cylindrical":

        # Compute length of air column first
        lengths = compute_length_of_air_column_cylindrical(t, **params)

        if vibration_type == "axial":
            frequencies = compute_axial_frequency_cylindrical(lengths, **params)

            if harmonic is not None:
                assert harmonic > 0 and isinstance(harmonic, int)
                frequencies = frequencies * harmonic
            
        elif vibration_type == "radial":
            if harmonic is None:
                mode = 1
            else:
                assert isinstance(harmonic, int)
                assert harmonic in [1, 2]
                mode = harmonic + 1
            frequencies = compute_radial_frequency_cylindrical(
                lengths, mode=mode, **params,
            )

        else:
            raise NotImplementedError
    
    elif container_shape == "bottleneck":

        # Compute length of air column first assuming 
        # base of the bottle is a cylindrical
        lengths = compute_length_of_air_column_cylindrical(t, **params)

        if vibration_type == "axial":
            frequencies = compute_axial_frequency_bottleneck(
                lengths, **params,
            )

            if harmonic is not None:
                assert harmonic > 0 and isinstance(harmonic, int)
                frequencies = frequencies * harmonic
        else:
            raise NotImplementedError
    
    else:
        raise ValueError
    
    return frequencies
