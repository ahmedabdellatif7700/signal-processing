import numpy as np
from typing import Tuple

class TxDSP:
    def __init__(self):
        self._reset()

    def _reset(self) -> None:
        """Reset all attributes to default values."""
        self.P = {
            'format': 0,
            'Rb': 0.0,
            'fs': 0.0,
            'M': 0,
            'k': 0,
            'Nbits': 0,
            'Nsym': 0,
            'Rs': 0.0,
            'Ts': 0.0,
            'ts': 0.0,
            'sps': 0,
            'symbols': np.array([], dtype=np.complex64),
            'OriginalSymbols': np.array([], dtype=np.complex64),
        }

    def configure(self, Nbits: int, format: int = 1, Rb: float = 100e9, fs: float = 8*100e9) -> None:
        """
        Configure TxDSP for QPSK.
        Args:
            Nbits: Number of bits to transmit.
            format: Modulation format (only 1 for QPSK supported).
            Rb: Bit rate (default: 100 Gbps).
            fs: Sampling frequency (default: 8× bit rate).
        """
        self.P['format'] = format
        self.P['Rb'] = Rb
        self.P['fs'] = fs
        self.P['Nbits'] = Nbits
        if self.P['format'] != 1:
            raise ValueError("Only QPSK (format=1) is supported.")
        # QPSK parameters
        self.P['M'] = 4
        self.P['k'] = 2
        self.P['Rs'] = self.P['Rb'] / self.P['k']
        self.P['Ts'] = 1.0 / self.P['Rs']
        self.P['ts'] = 1.0 / self.P['fs']
        self.P['sps'] = round(self.P['fs'] / self.P['Rs'])
        self.P['Nsym'] = self.P['Nbits'] // self.P['k']

    def _bits_to_symbols(self, bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Private method: Map bits to QPSK symbols using Gray coding.
        Args:
            bits: Input bits as a numpy array.
        Returns:
            Tuple of (normalized_symbols, original_symbols).
        """
        if len(bits) % self.P['k'] != 0:
            raise ValueError("Bits must be divisible by k.")
        B = bits.reshape(-1, self.P['k'])
        symbols = np.empty(B.shape[0], dtype=np.complex64)
        # Gray-coded mapping: (b1,b2) -> QPSK symbol
        mapping = {
            (0, 0): 1 + 1j,
            (0, 1): -1 + 1j,
            (1, 1): -1 - 1j,
            (1, 0): 1 - 1j
        }
        for i in range(B.shape[0]):
            b1, b2 = B[i, 0], B[i, 1]
            symbols[i] = mapping[(b1, b2)]
        original = symbols.copy()
        normalized = symbols / np.max(np.abs(symbols))
        return normalized, original

    def generate_signal(self, bits: np.ndarray) -> Tuple[None, np.ndarray]:
        """
        Generate QPSK symbols and return as a tuple for backward compatibility.
        Args:
            bits: Input bits as a numpy array.
        Returns:
            Tuple of (None, normalized_symbols) for backward compatibility.
        """
        Tx_s, OriginalSymbols = self._bits_to_symbols(bits)
        self.P['symbols'] = Tx_s
        self.P['OriginalSymbols'] = OriginalSymbols
        return None, Tx_s  # Return (None, normalized_symbols) to match existing code
