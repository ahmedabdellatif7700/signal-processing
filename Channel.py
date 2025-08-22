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




    def __init__(self):
        pass

    def configure(self, choice=1, awgn=True):
        """Configure channel type and whether to add AWGN."""
        if choice not in self._channel_responses:
            raise ValueError("Invalid channel choice. Must be 1-6.")
        self._choice = choice
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
        # for debugging only .. this syntax is correct and is aligned 
        # # For now, just pass tx_symbols directly
        # rx_symbols_1 = tx_symbols.astype(np.complex64)

        # if np.allclose(rx_symbols_1, rx_symbols):
        #     print("They match")
        # else:
        #     print("They differ")


        # Add AWGN if flag is True
        if self._awgn:
            snr_linear = 10 ** (snr_db / 10)
            power = np.mean(np.abs(rx_symbols) ** 2)
            noise_std = np.sqrt(power / (2 * snr_linear)) if power > 0 else 0.0
            noise = noise_std * (np.random.randn(len(rx_symbols)) + 1j * np.random.randn(len(rx_symbols)))
            rx_symbols += noise.astype(np.complex64)

        self._rx_symbols = rx_symbols
        return self, rx_symbols
    
    # getter for impluse response
    def get_impulse_response(self):
        """Return a copy of the current impulse response (keeps _impulse_response private)."""
        return self._impulse_response.copy()

