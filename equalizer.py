import numpy as np

def adaptive_equalizer_qam(x, a, P):
    """
    Adaptive equalization at 2SPS for QAM signals.
    Parameters:
    - x: 2D array of received signals (each column is a signal) at 2SPS.
    - a: 2D array of transmitted training sequences.
    - P: Dictionary of equalizer parameters.
    Returns:
    - y: 2D array of processed signals (each column is a signal) at 2SPS.
    - w: Equalizer taps (2D array).
    - e: Equalizer error (2D array).
    """
    # Retrieve parameters
    Ntaps = P['Ntaps']  # Taps of the adaptive equalizer (must be even)
    nSpS = P['nSpS']    # Number of samples per symbol
    mus = P['mus']      # Adaptation coefficients
    methods = P['methods']  # Equalizer algorithms (e.g., ['lms', 'lms_dd'])
    eqmode = P['eqmode']    # Equalizer type (e.g., 'FFE')
    Ks = P['Ks']        # Number of symbols to run the equalizer
    C = P['C']          # QAM constellation

    # Check input parameters
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        raise ValueError("x must be a 2D array")
    if not isinstance(a, np.ndarray) or a.ndim != 2:
        raise ValueError("a must be a 2D array")
    if Ntaps % 2 != 0:
        raise ValueError("Ntaps must be even")
    if len(mus) != len(methods):
        raise ValueError("mus and methods must have the same length")
    if len(Ks) != len(methods):
        raise ValueError("Ks and methods must have the same length")

    # Precalculate parameters
    Lx = x.shape[0]     # Number of samples to process
    Nrx = x.shape[1]    # Number of received signals
    Ntx = a.shape[1]    # Number of transmitted signals

    # Initialize outputs as complex arrays
    y = np.zeros((Lx, Ntx), dtype=complex)  # Output signal (complex)
    e = np.zeros((Lx, Ntx), dtype=complex)  # Error signal (complex)
    w = np.zeros((Nrx * Ntaps, Ntx), dtype=complex)  # Equalizer taps (complex)

    # Run the correct equalizer
    if eqmode == 'FFE':
        y, w, e = eq_FFE(x, Lx, Nrx, a, Ntx, mus, Ntaps, nSpS, Ks, methods, C)
    else:
        raise ValueError(f"Equalizer mode {eqmode} not supported")

    # Cut head and tail
    y = y[Ntaps:Lx - Ntaps - 1, :]
    return y, w, e

def eq_FFE(x, Lx, Nrx, a, Ntx, mus, Ntaps, nSpS, Ks, methods, C):
    """
    Complex-valued Data-Aided/Decision-Directed FFE for QAM.
    """
    Lpk = Ntaps // 2  # Peak position in equalizer taps
    vec = np.arange(-Lpk, Lpk)  # Preallocate data vector
    fin = 0  # Initialize end index

    # Initialize as complex arrays
    y = np.zeros((Lx, Ntx), dtype=complex)  # Output signal (complex)
    e = np.zeros((Lx, Ntx), dtype=complex)  # Error signal (complex)
    w = np.zeros((Nrx * Ntaps, Ntx), dtype=complex)  # Equalizer taps (complex)

    for m in range(len(methods)):
        iniz = max(Ntaps, fin + 1)  # Starting point of equalizer
        fin = min(Lx - Ntaps - 1, Ks[m])  # Ending point of equalizer
        mu = mus[m]  # Adaptive coefficient

        for nn in range(iniz, fin + 1):
            # Apply equalizer
            u = x[nn + vec, :].reshape(Nrx * Ntaps, 1).flatten()  # u is complex
            y[nn, :] = np.dot(u, w)  # y is complex

            # Calculate error every nSpS samples
            if nn % nSpS == 0:
                if methods[m] == 'lms':
                    i_a = (nn // nSpS - 1) % a.shape[0]  # Index of training symbol
                    adec = a[i_a, :]  # adec is complex (QPSK symbols)
                elif methods[m] == 'lms_dd':
                    adec = qam_hard_decision(y[nn, :], C)  # adec is complex
                else:
                    raise ValueError(f"Method {methods[m]} not supported")

                e[nn, :] = y[nn, :] - adec  # e is complex
                w -= mu * np.outer(u, e[nn, :])  # Update complex weights

    # Cut head and tail
    y = y[Ntaps:Lx - Ntaps - 1, :]
    return y, w, e

def qam_hard_decision(x, C):
    """
    QAM hard decision: maps input to the nearest constellation point.
    Returns complex values.
    """
    x = np.atleast_1d(x)
    distances = np.abs(C - x[:, np.newaxis]) ** 2
    d = C[np.argmin(distances, axis=1)]
    return d
