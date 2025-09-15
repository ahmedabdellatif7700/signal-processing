import numpy as np
from scipy.signal import lfilter


# ======================= TxDSP
class TxDSP:
    def __init__(self):
        pass

    def modulate(
        self,
        x,
        M,
        symbolOrderStr="bin",
        symbolOrderVector=None,
        bitInput=False,
        unitAveragePower=False,
        outputDataType=None,
    ):
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
                outputDataType = (
                    np.complex128 if x.dtype == np.float64 else np.complex64
                )

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
            if symbolOrder.lower() == "custom":
                if symbolOrderVector is None:
                    raise ValueError(
                        "symbolOrderVector must be provided for 'custom' ordering."
                    )
                symbolOrderMap = np.zeros(M, dtype=x.dtype)
                symbolOrderMap[symbolOrderVector] = np.arange(M)
                msg = symbolOrderMap[x]
            elif symbolOrder.lower() == "gray":
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
            sym = np.array([int("".join(map(str, bits)), 2) for bits in x_reshaped])
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


# ======================= Channel
class Channel:
    """Linear/nonlinear channel with optional AWGN."""
    _responses = {1: np.array([1.0,0,0], dtype=float),
                  2: np.array([0.447,0.894,0], dtype=float)}

    def configure(self, choice, nl, awgn):
        self._choice = choice
        self._nl = nl
        self._awgn = awgn
        self._impulse_response = self._responses[choice]
        print(f"Channel configured: choice={choice}, NL={nl}, impulse_response={self._impulse_response}")
        return self

    def apply_channel(self, t_k , snr_db):
        r_k = lfilter(self._impulse_response, [1.0], t_k)
        if self._nl==1: 
            r_k = np.tanh(r_k)
        elif self._nl==2:
            # rx = rx + 0.2*rx**2 - 0.1*rx**3
            pass
        if self._awgn:
            snr_lin = 10**(snr_db/10)
            noise_std = np.sqrt(np.mean(np.abs(r_k)**2)/(2*snr_lin))
            r_k += noise_std*(np.random.randn(len(r_k))+1j*np.random.randn(len(r_k)))
        return r_k

        # if self._awgn:
        #     rx = self.add_awgn(rx, snr_db)
        # return rx

    def add_awgn(self, x, reqSNR, sigPower="measured", powerType="db", seed=None
    ):
        """
        Internal method to add white Gaussian noise to the signal.
        """
        # Handle signal power
        if sigPower == "measured":
            sigPower = np.mean(np.abs(x) ** 2)
        elif sigPower == "default":
            sigPower = 1.0  # 0 dBW
        else:
            sigPower = float(sigPower)

        # Convert SNR to linear scale if needed
        if powerType.lower() == "db":
            reqSNR = 10 ** (np.array(reqSNR) / 10)
            if isinstance(sigPower, str) and sigPower not in (
                "measured",
                "default",
            ):
                sigPower = 10 ** (float(sigPower) / 10)
        elif powerType.lower() != "linear":
            raise ValueError("powerType must be 'db' or 'linear'")

        # Validate SNR
        if np.any(np.array(reqSNR) < 0):
            raise ValueError("SNR must be non-negative")

        # Ensure both are numeric
        sigPower = float(sigPower)
        reqSNR = np.array(reqSNR, dtype=float)

        # Compute noise power
        noisePower = sigPower / reqSNR

        # Generate noise
        if seed is not None:
            rng = np.random.RandomState(seed)
            noise = (
                rng.randn(*x.shape) + 1j * rng.randn(*x.shape)
                if np.iscomplexobj(x)
                else rng.randn(*x.shape)
            )
        else:
            noise = (
                np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)
                if np.iscomplexobj(x)
                else np.random.randn(*x.shape)
            )

        # Scale noise
        noise = np.sqrt(noisePower) * noise

        # Add noise to signal
        y = x + noise

        # Compute noise variance
        var = noisePower
        return y, var


# ======================= SysOrch
class SysOrch:
    def __init__(self):
        # Instantiate DSP objects here
        self.tx_dsp = TxDSP()
        self.channel = Channel().configure(choice=1, nl=0, awgn=True)

    def run(self):
        print("Running SysOrch")
        M = 16
        d = np.random.randint(0, M, size=1000)  # random integers between 0 and M-1

        # 16-QAM modulation with Gray ordering, treating input as integers
        x, const = self.tx_dsp.modulate(d, M, "gray", bitInput=False)

        print("Modulated output (first 10):", x[:10])
        print("Constellation points:", const)

    
        # Apply channel and AWGN
        r_k = self.channel.apply_channel(x, snr_db=10)
        print("Received signal (first 10):", r_k[:10])


if __name__ == "__main__":
    sim = SysOrch()
    sim.run()
