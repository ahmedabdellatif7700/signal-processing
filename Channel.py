import numpy as np
from scipy.signal import lfilter

class Channel:
    # Predefined channel impulse responses
    _channel_responses = {
        1: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        2: np.array([0.447, 0.894, 0.0], dtype=np.float32),
        3: np.array([0.209, 0.995, 0.209], dtype=np.float32),
        4: np.array([0.260, 0.930, 0.260], dtype=np.float32),
        5: np.array([0.304, 0.903, 0.304], dtype=np.float32),
        6: np.array([0.341, 0.876, 0.341], dtype=np.float32),
    }
    # ===================== %% NL BLOCK ==============================
    # The nonlinear channel output is modeled as:
    # a'(k) = ϕ( t(k), t(k-1), ..., t(k-Nh+1); h(0), h(1), ..., h(Nh-1) )
    #
    # Nonlinearity modes (NL):
    # NL=0: b(k) = a(k)                              (linear channel)
    # NL=1: b(k) = tanh(a(k))                        (tanh nonlinearity)
    # NL=2: b(k) = a(k) + 0.2*a(k)^2 - 0.1*a(k)^3   (polynomial nonlinearity)
    # NL=3: b(k) = a(k) + 0.2*a(k)^2 - 0.1*a(k)^3 + 0.5*cos(pi*a(k)) (poly + cosine)


    def __init__(self):
        pass

    def configure(self, choice=1, nl = 0,awgn=True):
        """Configure channel type and whether to add AWGN."""
        if choice not in self._channel_responses:
            raise ValueError("Invalid channel choice. Must be 1-6.")
        self._choice = choice
        self._nl = nl
        self._awgn = awgn
        self._impulse_response = self._channel_responses[self._choice]
        # self._write_config()
        return self

    # def _write_config(self):
    #     """Optional: write configuration to external params object if it exists."""
        # if self._params is not None:
        #     self._params.channel_choice = self._choice
        #     self._params.awgn = self._awgn
        #     self._params.impulse_response = self._impulse_response.copy()

    def apply_channel(self, tx_symbols, snr_db):
        """Apply channel impulse response and optional AWGN to transmitted symbols."""
        if len(tx_symbols) == 0:
            raise ValueError("Transmitted symbols are empty.")

        # Apply channel (convolution)
        rx_symbols = lfilter(
            self._impulse_response.astype(np.complex64),  # b (FIR taps)
            [1.0],                                        # a (IIR denominator = 1 → FIR)
            tx_symbols.astype(np.complex64)               # input
        )

        rx_symbols = np.asarray(rx_symbols, dtype=np.complex64)  # force ndarray

        # Check if tx and rx match exactly
        if np.array_equal(tx_symbols, rx_symbols):
            print("tx_symbols and rx_symbols match.")

        # ===================== %% NL BLOCK ==============================
        if self._nl == 0:
            # Linear channel, do nothing
            rx_symbols_nl = rx_symbols
        elif self._nl == 1:
            # tanh nonlinearity
            rx_symbols_nl = np.tanh(rx_symbols)
            print("tanh nonlinearity")
        elif self._nl == 2:
            # Polynomial nonlinearity
            rx_symbols_nl = rx_symbols + 0.2 * rx_symbols**2 - 0.1 * rx_symbols**3
            print("Polynomial nonlinearity")
        elif self._nl == 3:
            # Polynomial + cosine
            rx_symbols_nl = rx_symbols + 0.2 * rx_symbols**2 - 0.1 * rx_symbols**3 + 0.5 * np.cos(np.pi * rx_symbols)
            print("Polynomial + cosine error")
        else:
            raise ValueError("Invalid NL choice. Must be 0-3.")
        

        # Add AWGN if flag is True
        if self._awgn:
            snr_linear = 10 ** (snr_db / 10)
            power = np.mean(np.abs(rx_symbols_nl) ** 2)
            noise_std = np.sqrt(power / (2 * snr_linear)) if power > 0 else 0.0
            noise = noise_std * (np.random.randn(len(rx_symbols_nl)) + 1j * np.random.randn(len(rx_symbols_nl)))
            rx_symbols_nl += noise.astype(np.complex64)

        self.rx_symbols_nl = rx_symbols_nl
        return self, rx_symbols_nl
    
    # getter for impluse response
    def get_impulse_response(self):
        """Return a copy of the current impulse response (keeps _impulse_response private)."""
        return self._impulse_response.copy()
    
    def get_channel_choice(self):
        return self._choice