import numpy as np
from equalizer import adaptive_equalizer_qam

class RxDSP:
    def __init__(self):
        self._reset()

    def _reset(self):
        """Reset all private attributes."""
        self._format = "QPSK"
        self._ber = np.float32(0.0)
        self._decoded_bits = np.array([], dtype=np.int32)
        self._C = np.array([1+1j, -1+1j, -1-1j, 1-1j])  # QPSK constellation
        self._B = np.array([[0,0],[0,1],[1,1],[1,0]])  # Gray-coded bits
        self._eq_type = "None"

    def configure(self, eq_type="None"):
        """Configure equalizer type."""
        self._eq_type = eq_type

    def normalize_signal(self, r_k):
        """Remove DC offset and normalize by standard deviation."""
        r_k = r_k - np.mean(r_k, axis=0, keepdims=True)
        return r_k / np.std(r_k, axis=0, keepdims=True)

    def select_training(self, t_k):
        """Select training sequence (placeholder: full QPSK)."""
        return t_k

    def _cross_correlation_align(self, r_k, t_k):
        """Align received signal to transmitted training sequence."""
        corr = np.correlate(r_k.flatten(), t_k.flatten(), mode='full')
        delay = np.argmax(np.abs(corr)) - (len(t_k) - 1)
        if delay >= 0:
            r_aligned = r_k[delay:]
            t_aligned = t_k[:len(r_aligned)]
        else:
            r_aligned = r_k[-delay:]
            t_aligned = t_k[:len(r_aligned)]
        return r_aligned, t_aligned

    def _adaptive_equalizer(self, r_k, t_k, P):
        """LMS feed-forward equalizer (FFE) for QAM."""
        y, w, e = adaptive_equalizer_qam(r_k, t_k, P)
        return y, w, e

    def _min_distance_decision(self, r_k):
        """Decode symbols and compute SNR (placeholder)."""
        r_k = np.atleast_1d(r_k)  # Ensure 1D
        yd = np.zeros_like(r_k, dtype=complex)
        for i, sym in enumerate(r_k):
            idx = np.argmin(np.abs(sym - self._C))
            yd[i] = self._C[idx]
        snr = 0.0  # Replace with actual SNR calculation
        return yd
    

    # def _min_distance_decision(self, r_k):
    #     """Minimum-distance decoding for QPSK."""
    #     yd = np.zeros(len(r_k), dtype=complex)
    #     for i, sym in enumerate(r_k):
    #         idx = np.argmin(np.abs(sym - self._C))
    #         yd[i] = self._C[idx]
    #     return yd 

    def _symbols_to_bits(self, yd):
        bits = []
        for sym in yd:
            idx = np.argmin(np.abs(sym - self._C))
            bits.extend(self._B[idx])
        return np.array(bits, dtype=int)

    def process_signal(self, t_k, r_k, P):
        """
        Process QPSK signal.
        Inputs:
            t_k (1D/2D): Transmitted symbols
            r_k (1D/2D): Received symbols
            P (dict): Parameters for processing
        """
        # Ensure 2D inputs
        t_k = np.atleast_2d(t_k).T
        r_k = np.atleast_2d(r_k).T

        # Cut transmitted sequence
        t_k = t_k[P["init_cut"]:]

        # Normalize received signal
        r_k = self.normalize_signal(r_k)

        # Align the sequence
        t_train = self.select_training(t_k)
        r_k, t_train = self._cross_correlation_align(r_k, t_train)

        # Resampling logic
        SampInst = P["ovsa"]
        nsps = P["ovsa"]
        BERss, opt_mus = [], []

        for init_sample in range(SampInst):
            # Decimate/resample
            x_curr = r_k[init_sample::nsps, :]
            t_curr = np.tile(t_train, (1, int(np.ceil(len(x_curr) / len(t_train))))).T[:len(x_curr), :]
            BERs, decoded_bits_per_mu = [], []

            for mu in P["mus"]:
                if self._eq_type == "FFE":
                    eq_P = {
                        "Ntaps": P["taps"],
                        "nSpS": P["nSpS"],
                        "mus": np.array([mu]),
                        "methods": P["methods"],
                        "eqmode": self._eq_type,
                        "Ks": np.array([len(x_curr)]),
                        "C": self._C
                    }
                    y, w, e = self._adaptive_equalizer(x_curr, t_curr, eq_P)
                    y_ds = y[::P["nSpS"], :]  # 2D array (e.g., (N, 1))
                    y_ds = y_ds[P["Ks"][0]+1:, :].flatten()  # Convert to 1D and remove training tails
                    start_idx = P["Kcut"] + P["taps"]*2 + 1
                    end_idx = start_idx + len(y_ds)
                    if end_idx > len(t_curr):
                        end_idx = len(t_curr)
                        y_ds = y_ds[:end_idx - start_idx]
                    t_eval = t_curr[start_idx:end_idx, :].flatten()  # 1D slice
                else:
                    y_ds = x_curr.flatten()
                    t_eval = t_curr.flatten()

                # Decode symbols
                yd = self._min_distance_decision(y_ds)
                decoded_bits = self._symbols_to_bits(yd)
                t_bits = self._symbols_to_bits(t_eval)

                # BER computation
                min_len = min(len(decoded_bits), len(t_bits))
                ber = np.sum(np.abs(decoded_bits[:min_len] - t_bits[:min_len])) / min_len
                BERs.append(ber)
        
                decoded_bits_per_mu.append(decoded_bits)

            # Pick best mu for this sampling instant
            best_idx = np.argmin(BERs) 
            BERss.append(BERs[best_idx])

            opt_mus.append(mu)

            if P["disp_out"]:
                print(f"Sampling instant {init_sample+1}/{SampInst} | BER={BERs[best_idx]:.4e}")

        # Pick best sampling instant
        best_samp_idx = np.argmin(BERss)
        self._ber = np.float32(BERss[best_samp_idx])
        self._decoded_bits = decoded_bits_per_mu[best_samp_idx]
        self.opt_mu = opt_mus[best_samp_idx]

        if P["disp_out"]:
            print(f"Final selected BER={self._ber:.4e}, mu={self.opt_mu:.4e}")

        return self._ber, self.opt_mu, P
