import numpy as np

class TxDSP:
    def __init__(self):
        self._reset()

    def _reset(self):
        self._format = np.int32(0)
        self._Rb = np.float32(0.0)
        self._fs = np.float32(0.0)
        self._M = np.int32(0)
        self._k = np.int32(0)
        self._a = np.float32(0.0)
        self._Nbits = np.int32(0)
        self._Nsym = np.int32(0)
        self._Rs = np.float32(0.0)
        self._Ts = np.float32(0.0)
        self._ts = np.float32(0.0)
        self._sps = np.int32(0)
        self._symbols = np.array([], dtype=np.complex64)

    def configure(self, Nbits, format=1, Rb=100e9, fs=8*100e9):
        """Configure TxDSP for QPSK and write configuration to external params object."""
        self._format = np.int16(format)
        self._Rb = np.float32(Rb)
        self._fs = np.float32(fs)
        self._Nbits = np.int32(Nbits)

        if self._format != 1:
            raise ValueError("Only QPSK (format=1) is supported.")

        # QPSK parameters
        self._M = np.int32(4)
        self._k = np.int32(2)
        self._fs = np.float32(8 * Rb)
        self._Rs = np.float32(self._Rb / self._k)
        self._Ts = np.float32(1.0 / self._Rs)
        self._ts = np.float32(1.0 / self._fs)
        self._sps = np.int16(round(self._fs / self._Rs))
        self._Nsym = np.int32(self._Nbits // self._k)
        self._a = np.float32(1.0 / np.sqrt(2.0))

        # def _write_config(self,params):
        #     """Optional placeholder if you want to store config somewhere."""
        # attrs = ['format', 'M', 'k', 'Rb', 'fs', 'Rs', 'Ts', 'ts', 'sps', 'Nsym', 'a']
        # for attr in attrs:
        #     setattr(params, attr, getattr(self, f'_{attr}'))
        # return self

    def generate_signal(self, bits):
        """Generate QPSK symbols using Gray coding. Returns symbols without touching params."""
        if len(bits) % self._k != 0:
            raise ValueError("Bits must be divisible by k.")

        B = bits.reshape(-1, self._k)
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
            symbols[i] = self._a * mapping[(b1, b2)]

        self._symbols = symbols
        return self, symbols
