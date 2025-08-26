import numpy as np

class RxDSP:
    def __init__(self):
        self._reset()

    def _reset(self):
        """Reset all private attributes to default values."""
        self._format = "QPSK"
        self._ber = np.float32(0.0)
        self.decoded_bits = np.array([], dtype=np.int32)  # renamed from _B_hat
        # QPSK Gray-coded constellation
        self._C = np.array([1+1j, -1+1j, -1-1j, 1-1j])
        self._B = np.array([[0,0],[0,1],[1,1],[1,0]])

    def configure(self):
        pass

    def normalize_signal(self, r_k):
        """RMS normalize received signal."""
        if len(r_k) == 0:
            raise ValueError("Received signal is empty.")
        return r_k / np.sqrt(np.mean(np.abs(r_k)**2))

    def _cross_correlation_align(self, r_k, t_k):
        """Align received signal to transmitted sequence using cross-correlation."""
        corr = np.correlate(r_k, t_k, mode='full')
        delay = np.argmax(np.abs(corr)) - (len(t_k) - 1)
        if delay >= 0:
            r_aligned = r_k[delay:]
            t_aligned = t_k[:len(r_aligned)]
        else:
            r_aligned = r_k[-delay:]
            t_aligned = t_k[:len(r_aligned)]
        return r_aligned, t_aligned

    def _min_distance_decision(self, r_k):
        """Minimum-distance decoding for QPSK."""
        yd = np.zeros(len(r_k), dtype=complex)
        for i, sym in enumerate(r_k):
            idx = np.argmin(np.abs(sym - self._C))
            yd[i] = self._C[idx]
        return yd

    def _symbols_to_bits(self, yd):
        """Map decided symbols to bits using Gray-coded mapping."""
        bits = []
        for sym in yd:
            idx = np.argmin(np.abs(sym - self._C))
            bits.extend(self._B[idx])
        return np.array(bits, dtype=int)

    def process_signal(self, t_k, r_k, choice):
        """
        Process received QPSK signal, align, decode, and compute BER.

        Parameters
        ----------
        t_k : np.ndarray
            Transmitted QPSK symbols (complex).
        r_k : np.ndarray
            Received QPSK symbols (complex).
        choice : int
            Channel choice (1 → no cut, 2-6 → apply InitCut).
        """
        if len(t_k) == 0 or len(r_k) == 0:
            raise ValueError("Transmitted or received signal is empty.")

        # RMS normalize received signal
        r_k = self.normalize_signal(r_k)

        # Align received to transmitted sequence using cross-correlation
        r_k, t_k = self._cross_correlation_align(r_k, t_k)

        # Decode symbols using minimum-distance
        yd = self._min_distance_decision(r_k)
        self.decoded_bits = self._symbols_to_bits(yd)  # consistent naming

        # Convert transmitted symbols to bits
        t_bits = self._symbols_to_bits(t_k)

        # Compute BER
        min_len = min(len(t_bits), len(self.decoded_bits))
        errors = np.sum(np.abs(t_bits[:min_len] - self.decoded_bits[:min_len]))
        self._ber = np.float32(errors / min_len)

        return self, self._ber
