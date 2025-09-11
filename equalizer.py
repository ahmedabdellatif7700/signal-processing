import numpy as np

class LinearEqualizer:
    """
    Linear equalizer with LMS (can be extended to RLS/CMA) and
    hybrid supervised -> decision-directed adaptation.
    """

    def __init__(self,
                 num_taps=8,
                 algorithm='LMS',
                 step_size=0.01,
                 forgetting_factor=0.3,
                 delta=1e3,
                 reference_tap=None,
                 constellation=None):
        self.num_taps = int(num_taps)
        self.algorithm = algorithm.upper()
        assert self.algorithm in ('LMS', 'RLS', 'CMA'), "Algorithm must be 'LMS', 'RLS', or 'CMA'"
        self.step_size = float(step_size)
        self.forgetting_factor = float(forgetting_factor)
        self.delta = float(delta)
        if reference_tap is None:
            self.reference_tap = (self.num_taps // 2)
        else:
            self.reference_tap = max(0, int(reference_tap) - 1)
        self.constellation = None if constellation is None else np.array(constellation, dtype=complex)
        self.reset()

        # Print configuration
        print("=== LinearEqualizer Configuration ===")
        print(f"Algorithm: {self.algorithm}")
        print(f"NumTaps: {self.num_taps}")
        print(f"StepSize / ForgettingFactor: {self.step_size} / {self.forgetting_factor}")
        print(f"ReferenceTap (0-based): {self.reference_tap}")
        if self.constellation is not None:
            print(f"Constellation: {self.constellation}")
        print("====================================\n")

    def reset(self):
        """Initialize weights and RLS matrix if needed."""
        self.weights = np.zeros(self.num_taps, dtype=complex)
        self.weights[self.reference_tap] = 1.0 + 0j
        self.P = (self.delta * np.eye(self.num_taps, dtype=complex)) if self.algorithm == 'RLS' else None

    def _decision(self, y):
        """Decision-directed mapping to nearest constellation point."""
        if self.constellation is not None:
            dists = np.abs(y.reshape(-1, 1) - self.constellation.reshape(1, -1))
            idx = dists.argmin(axis=1)
            return self.constellation[idx]
        else:
            return np.sign(y.real) + 1j*np.sign(y.imag)

    def _compute_cma_R(self):
        if self.constellation is None:
            return 1.0
        magsq = np.abs(self.constellation)**2
        return (np.mean(magsq**2) / np.mean(magsq))

    def equalize(self, x, d=None, Ks=None):
        """
        Equalize a sequence with optional supervised training and decision-directed mode.

        Parameters
        ----------
        x : array_like
            Received complex samples (1D)
        d : array_like, optional
            Desired sequence for supervised training
        Ks : int, optional
            Number of supervised training symbols. After Ks, decision-directed mode is used.

        Returns
        -------
        y_out : np.ndarray
            Equalizer output sequence
        e_out : np.ndarray
            Error sequence used for adaptation
        """
        x = np.asarray(x, dtype=complex).ravel()
        if d is not None:
            d = np.asarray(d, dtype=complex).ravel()
        N = x.size
        pad = np.zeros(self.num_taps - 1, dtype=complex)
        buffer = np.concatenate([pad, x])
        y_out = np.empty(N, dtype=complex)
        e_out = np.empty(N, dtype=complex)

        if Ks is None:
            Ks = N if d is not None else 0
        Ks = int(Ks)

        print(f"Starting equalization: {N} samples, Ks={Ks} supervised samples\n")

        for n in range(N):
            u = buffer[n : n + self.num_taps][::-1]  # newest first
            y = np.vdot(self.weights, u)

            # Determine error
            if (d is not None) and (n < Ks):
                desired = d[n]
                err = desired - y
                mode = "Training"
            else:
                if self.constellation is None:
                    err = 0.0 + 0j
                    mode = "No adaptation (no constellation)"
                else:
                    decided = self._decision(np.array([y]))[0]
                    err = decided - y
                    mode = "Decision-Directed"

            # Update weights (LMS only for now)
            if self.algorithm == "LMS":
                self.weights += self.step_size * np.conj(err) * u

            y_out[n] = y
            e_out[n] = err

            # Debug print every 100 samples
            if n % max(1, N//10) == 0:
                print(f"Sample {n+1}/{N}: y={y:.4f}, err={err:.4f}, mode={mode}")
                print(f"Current weights (center 3 taps): {self.weights[max(0,n-1):min(self.num_taps,n+2)]}\n")

        print("Equalization completed.\n")
        return y_out, e_out
