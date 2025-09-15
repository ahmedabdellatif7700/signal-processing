import numpy as np
from scipy.signal import lfilter
from scipy.linalg import toeplitz
import matplotlib.pylab as plt

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
        def get_constellation(M, unitAveragePower, outputDataType):
            if outputDataType is None:
                outputDataType = (
                    np.complex128 if x.dtype == np.float64 else np.complex64
                )
            if unitAveragePower:
                const = get_unit_power_constellation(M, outputDataType)
            else:
                const = get_standard_constellation(M, outputDataType)
            if x.ndim > 1 and x.shape[0] > 1 and x.shape[1] == 1:
                const = const.reshape(-1, 1)
            return const

        def get_standard_constellation(M, dtype):
            k = int(np.log2(M))
            if k % 2 != 0:
                raise ValueError("M must be a square power of 2 for QAM.")
            m = int(np.sqrt(M))
            real = np.arange(-(m - 1), m, 2)
            imag = np.arange(-(m - 1), m, 2)
            const = np.array([r + 1j * i for i in imag for r in real], dtype=dtype)
            return const

        def get_unit_power_constellation(M, dtype):
            const = get_standard_constellation(M, dtype)
            power = np.mean(np.abs(const) ** 2)
            const = const / np.sqrt(power)
            return const

        def process_symbols(x, M, symbolOrder, symbolOrderVector):
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
            else:
                msg = x
            return msg

        def bin2gray(x, M):
            gray = x ^ (x >> 1)
            return gray

        def process_bit_input(x, M, symbolOrder, symbolOrderVector, const):
            k = int(np.log2(M))
            if x.size % k != 0:
                raise ValueError(f"Input bit stream length must be a multiple of {k}.")
            x_reshaped = x.reshape(-1, k)
            sym = np.array([int("".join(map(str, bits)), 2) for bits in x_reshaped])
            msg = process_symbols(sym, M, symbolOrder, symbolOrderVector)
            y = const[msg]
            return y

        def process_int_input(x, M, symbolOrder, symbolOrderVector, const):
            msg = process_symbols(x, M, symbolOrder, symbolOrderVector)
            y = const[msg]
            return y

        const = get_constellation(M, unitAveragePower, outputDataType)

        if bitInput:
            y = process_bit_input(x, M, symbolOrderStr, symbolOrderVector, const)
        else:
            y = process_int_input(x, M, symbolOrderStr, symbolOrderVector, const)

        return y, const


# ======================= Channel
class Channel:
    _responses = {1: np.array([1.0, 0, 0], dtype=float),
                  2: np.array([0.447, 0.894, 0], dtype=float)}

    def configure(self, choice, nl, awgn):
        self._choice = choice
        self._nl = nl
        self._awgn = awgn
        self._impulse_response = self._responses[choice]
        print(f"Channel configured: choice={choice}, NL={nl}, impulse_response={self._impulse_response}")
        return self

    def apply_channel(self, t_k, snr_db):
        r_k = lfilter(self._impulse_response, [1.0], t_k)
        if self._nl == 1:
            r_k = np.tanh(r_k)
        elif self._nl == 2:
            pass
        if self._awgn:
            snr_lin = 10 ** (snr_db / 10)
            noise_std = np.sqrt(np.mean(np.abs(r_k) ** 2) / (2 * snr_lin))
            r_k += noise_std * (np.random.randn(len(r_k)) + 1j * np.random.randn(len(r_k)))
        return r_k

    def add_awgn(self, x, reqSNR, sigPower="measured", powerType="db", seed=None):
        if sigPower == "measured":
            sigPower = np.mean(np.abs(x) ** 2)
        elif sigPower == "default":
            sigPower = 1.0
        else:
            sigPower = float(sigPower)

        if powerType.lower() == "db":
            reqSNR = 10 ** (np.array(reqSNR) / 10)
            if isinstance(sigPower, str) and sigPower not in ("measured", "default"):
                sigPower = 10 ** (float(sigPower) / 10)
        elif powerType.lower() != "linear":
            raise ValueError("powerType must be 'db' or 'linear'")

        if np.any(np.array(reqSNR) < 0):
            raise ValueError("SNR must be non-negative")

        sigPower = float(sigPower)
        reqSNR = np.array(reqSNR, dtype=float)
        noisePower = sigPower / reqSNR

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

        noise = np.sqrt(noisePower) * noise
        y = x + noise
        var = noisePower
        return y, var


# ======================= LinearEqualizer
class LinearEqualizer:
    def __init__(self,
                 Algorithm='LMS',
                 NumTaps=5,
                 StepSize=0.01,
                 Constellation=None,
                 ReferenceTap=1,
                 InputDelay=0,
                 InputSamplesPerSymbol=1,
                 TrainingFlagInputPort=False,
                 AdaptAfterTraining=False,
                 AdaptWeightsSource='Property',
                 AdaptWeights=True,
                 InitialWeightsSource='Property',
                 InitialWeights=None,
                 BlindInitialWeights=None,
                 WeightUpdatePeriod=1):

        self.Algorithm = Algorithm
        self.NumTaps = NumTaps
        self.StepSize = StepSize
        self.Constellation = Constellation
        self.ReferenceTap = ReferenceTap
        self.InputDelay = InputDelay
        self.InputSamplesPerSymbol = InputSamplesPerSymbol
        self.TrainingFlagInputPort = TrainingFlagInputPort
        self.AdaptAfterTraining = AdaptAfterTraining
        self.AdaptWeightsSource = AdaptWeightsSource
        self.AdaptWeights = AdaptWeights
        self.InitialWeightsSource = InitialWeightsSource
        self.WeightUpdatePeriod = WeightUpdatePeriod

        if BlindInitialWeights is None:
            self.BlindInitialWeights = np.zeros(NumTaps, dtype=complex)
        else:
            self.BlindInitialWeights = np.asarray(BlindInitialWeights, dtype=complex)

        if InitialWeights is not None:
            self.weights = np.asarray(InitialWeights, dtype=complex)
        else:
            self.weights = self.BlindInitialWeights.copy()

        self.input_buffer = np.zeros(NumTaps, dtype=complex)
        self.symbol_counter = 0
        self.training_complete = False
        self._prev_training_flag = False

    @property
    def LocalNumForwardTaps(self):
        return self.NumTaps

    @property
    def LocalNumFeedbackTaps(self):
        return 0

    def reset(self):
        self.weights = self.BlindInitialWeights.copy()
        self.input_buffer = np.zeros(self.NumTaps, dtype=complex)
        self.symbol_counter = 0
        self.training_complete = False
        self._prev_training_flag = False

    def mmseweights(self, h, SNR):
        h = np.asarray(h, dtype=complex).flatten()
        num_taps = self.LocalNumForwardTaps
        h_len = len(h)

        delay = self.InputDelay
        total_delay = self.ReferenceTap + delay

        h_auto_corr = np.correlate(h, h, mode='full')
        r = np.concatenate([h_auto_corr[h_len-1:], np.zeros(num_taps - h_len, dtype=complex)])
        R = toeplitz(r[:num_taps], np.conj(r[:num_taps]))

        R += 0.5 * 10 ** (-SNR / 10) * np.eye(num_taps)
        p = np.zeros(num_taps, dtype=complex)

        if total_delay < h_len:
            p[:total_delay] = np.conj(h[:total_delay][::-1])
        else:
            p[total_delay - h_len:total_delay] = np.conj(h[::-1])

        w0 = np.linalg.solve(R, p)
        return np.conj(w0)

    def step(self, x, *args):
        x = np.asarray(x, dtype=complex).flatten()

        training_symbols = None
        training_flag = True
        adapt_weights = self.AdaptWeights

        if len(args) >= 1:
            training_symbols = np.asarray(args[0], dtype=complex).flatten()
        if self.TrainingFlagInputPort and len(args) == 2:
            training_flag = bool(args[1])
        elif not self.TrainingFlagInputPort and len(args) == 2:
            adapt_weights = bool(args[1])

        y = np.zeros(len(x), dtype=complex)
        e = np.zeros(len(x), dtype=complex)

        for i in range(len(x)):
            # Shift buffer: newest sample first
            self.input_buffer = np.roll(self.input_buffer, 1)
            self.input_buffer[0] = x[i]

            y[i] = np.dot(np.conj(self.weights), self.input_buffer)

            if adapt_weights and (self.symbol_counter % self.WeightUpdatePeriod == 0):
                if training_symbols is not None and i < len(training_symbols) and training_flag:
                    d = training_symbols[i]
                elif self.AdaptAfterTraining and self.Constellation is not None:
                    d = self.Constellation[np.argmin(np.abs(y[i] - self.Constellation))]
                else:
                    d = None

                if d is not None:
                    e[i] = d - y[i]
                    norm_factor = np.dot(self.input_buffer, np.conj(self.input_buffer)) + 1e-12
                    self.weights += (self.StepSize / norm_factor) * np.conj(e[i]) * self.input_buffer

            self.symbol_counter += 1

        if training_symbols is not None:
            self.training_complete = True

        return y, e, self.weights.copy()

    def isLocked(self):
        return self.training_complete

    def clone(self):
        return LinearEqualizer(
            Algorithm=self.Algorithm,
            NumTaps=self.NumTaps,
            StepSize=self.StepSize,
            Constellation=self.Constellation,
            ReferenceTap=self.ReferenceTap,
            InputDelay=self.InputDelay,
            InputSamplesPerSymbol=self.InputSamplesPerSymbol,
            TrainingFlagInputPort=self.TrainingFlagInputPort,
            AdaptAfterTraining=self.AdaptAfterTraining,
            AdaptWeightsSource=self.AdaptWeightsSource,
            AdaptWeights=self.AdaptWeights,
            InitialWeightsSource=self.InitialWeightsSource,
            InitialWeights=self.weights.copy(),
            BlindInitialWeights=self.BlindInitialWeights.copy(),
            WeightUpdatePeriod=self.WeightUpdatePeriod,
        )


# ======================= SysOrch
class SysOrch:
    def __init__(self):
        self.tx_dsp = TxDSP()
        self.channel = Channel().configure(choice=1, nl=0, awgn=True)

    def run(self):
        print("Running SysOrch")
        M = 16
        d = np.random.randint(0, M, size=1000)

        x, const = self.tx_dsp.modulate(d, M, "gray", bitInput=False)

        print("Modulated output (first 10):", x[:10])
        print("Constellation points:", const)

        r_k = self.channel.apply_channel(x, snr_db=40)
        print("Received signal (first 10):", r_k[:10])

        eq = LinearEqualizer(NumTaps=7, StepSize=0.01, Constellation=const, ReferenceTap=3)

        y, e, _ = eq.step(r_k, x[:100])

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(np.abs(e))
        plt.xlabel('Symbols')
        plt.ylabel('|e|')
        plt.title('Error Magnitude During Training')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.scatter(np.real(y), np.imag(y))
        plt.title('Equalized Signal Constellation')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

        y_rest, _, _ = eq.step(r_k[100:])
        print("Equalizer locked:", eq.isLocked())

        plt.figure()
        plt.scatter(np.real(np.concatenate([y, y_rest])), np.imag(np.concatenate([y, y_rest])))
        plt.title('Final Equalized Signal Constellation')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True)
        plt.axis('equal')
        plt.show()


if __name__ == "__main__":
    sim = SysOrch()
    sim.run()