import numpy as np
# ======================= TxDSP 
class TxDSP:
    def __init__(self):
        pass

    def modulate(self, x, M, symbolOrderStr='bin', symbolOrderVector=None, bitInput=False, unitAveragePower=False, outputDataType=None):
        """
        Quadrature amplitude modulation (QAM) of the message signal x.

        Parameters:
        - x: Input message signal (integers between 0 and M-1 if bitInput is False, otherwise binary).
        - M: Constellation order (must be an integer power of two).
        - symbolOrderStr: 'bin', 'gray', or 'custom'.
        - symbolOrderVector: Custom symbol order vector (only used if symbolOrderStr is 'custom').
        - bitInput: If True, x is treated as a bit stream.
        - unitAveragePower: If True, constellation is normalized to unit average power.
        - outputDataType: Data type of the output (e.g., np.complex64, np.complex128).

        Returns:
        - y: Modulated complex envelope.
        - const: QAM constellation used for modulation.
        """

        def get_constellation(M, unitAveragePower, outputDataType):
            """Generate QAM constellation."""
            if outputDataType is None:
                outputDataType = np.complex128 if x.dtype == np.float64 else np.complex64

            # Generate standard or unit-power constellation
            if unitAveragePower:
                const = get_unit_power_constellation(M, outputDataType)
            else:
                const = get_standard_constellation(M, outputDataType)

            # Reshape constellation to match input orientation
            if x.ndim > 1 and x.shape[0] > 1 and x.shape[1] == 1:  # Column vector
                const = const.reshape(-1, 1)
            return const

        def get_standard_constellation(M, dtype):
            """Generate standard QAM constellation."""
            k = int(np.log2(M))
            if k % 2 != 0:
                raise ValueError("M must be a square power of 2 for QAM.")
            m = int(np.sqrt(M))
            real = np.arange(-(m - 1), m, 2)
            imag = np.arange(-(m - 1), m, 2)
            const = np.array([r + 1j * i for i in imag for r in real], dtype=dtype)
            return const

        def get_unit_power_constellation(M, dtype):
            """Generate unit average power QAM constellation."""
            const = get_standard_constellation(M, dtype)
            power = np.mean(np.abs(const) ** 2)
            const = const / np.sqrt(power)
            return const

        def process_symbols(x, M, symbolOrder, symbolOrderVector):
            """Map input symbols to constellation indices."""
            if symbolOrder.lower() == 'custom':
                if symbolOrderVector is None:
                    raise ValueError("symbolOrderVector must be provided for 'custom' ordering.")
                symbolOrderMap = np.zeros(M, dtype=x.dtype)
                symbolOrderMap[symbolOrderVector] = np.arange(M)
                msg = symbolOrderMap[x]
            elif symbolOrder.lower() == 'gray':
                msg = bin2gray(x, M)
            else:  # 'bin'
                msg = x
            return msg

        def bin2gray(x, M):
            """Convert binary to Gray-coded indices."""
            k = int(np.log2(M))
            gray = x ^ (x >> 1)
            return gray

        def process_bit_input(x, M, symbolOrder, symbolOrderVector, const):
            """Process bit input: convert bits to symbols."""
            k = int(np.log2(M))
            if x.size % k != 0:
                raise ValueError(f"Input bit stream length must be a multiple of {k}.")
            x_reshaped = x.reshape(-1, k)
            sym = np.array([int(''.join(map(str, bits)), 2) for bits in x_reshaped])
            msg = process_symbols(sym, M, symbolOrder, symbolOrderVector)
            y = const[msg]
            return y

        def process_int_input(x, M, symbolOrder, symbolOrderVector, const):
            """Process integer input: map directly to constellation."""
            msg = process_symbols(x, M, symbolOrder, symbolOrderVector)
            y = const[msg]
            return y

        # Main function logic
        const = get_constellation(M, unitAveragePower, outputDataType)

        if bitInput:
            y = process_bit_input(x, M, symbolOrderStr, symbolOrderVector, const)
        else:
            y = process_int_input(x, M, symbolOrderStr, symbolOrderVector, const)

        return y, const

# ======================= SysOrch 
class SysOrch:
    def __init__(self):
        # Instantiate DSP objects here
        self.tx_dsp = TxDSP()

    def run(self):
        print("Running SysOrch")
        M = 16
        d = np.random.randint(0, M, size=1000)   # random integers between 0 and M-1

        # Example: 16-QAM modulation with Gray ordering, treating input as integers
        y, const = self.tx_dsp.modulate(d, M, 'gray', bitInput=False)

        print("Modulated output (first 10):", y[:10])
        print("Constellation points:", const)


if __name__ == "__main__":
    sim = SysOrch()
    sim.run()