"""
Implementation of the fixed, non-trainable OFDM Modulator and Demodulator.
These modules are based on the description in Section 5.2 of the Readme.md,
with correct handling for the null DC subcarrier.
"""

import torch
import torch.nn as nn
import torch.fft as fft

class OFDMModulator(nn.Module):
    """
    Performs OFDM modulation on a batch of resource grids.
    This includes mapping to FFT bins, performing IFFT, and adding a cyclic prefix.
    The DC subcarrier at index 0 is kept null.
    """
    def __init__(self, fft_size, cp_length, num_subcarriers):
        super().__init__()
        if num_subcarriers % 2 != 0:
            raise ValueError("Number of subcarriers must be even.")
        self.fft_size = fft_size
        self.cp_length = cp_length
        self.num_subcarriers = num_subcarriers

    def forward(self, x_grid):
        """
        Args:
            x_grid (torch.Tensor): Input resource grid of complex symbols.
                                  Shape: [batch_size, num_symbols, num_subcarriers]

        Returns:
            torch.Tensor: Serialized time-domain signal with cyclic prefix.
                          Shape: [batch_size, num_symbols * (fft_size + cp_length)]
        """
        batch_size, num_sym, _ = x_grid.shape
        
        x_fft = torch.zeros(
            batch_size, num_sym, self.fft_size, 
            dtype=x_grid.dtype, device=x_grid.device
        )
        
        # Split subcarriers around the DC bin
        half_sc = self.num_subcarriers // 2
        # Positive frequencies: indices 1 to half_sc
        x_fft[..., 1:half_sc + 1] = x_grid[..., half_sc:]
        # Negative frequencies: indices -half_sc to -1
        x_fft[..., -half_sc:] = x_grid[..., :half_sc]
        
        x_time = fft.ifft(x_fft, n=self.fft_size, dim=-1)
        
        cp = x_time[..., -self.cp_length:]
        x_time_cp = torch.cat([cp, x_time], dim=-1)
        
        return x_time_cp.reshape(batch_size, -1)

class OFDMDemodulator(nn.Module):
    """
    Performs OFDM demodulation on a batch of time-domain signals.
    This includes removing the cyclic prefix, performing FFT, and extracting active subcarriers
    while ignoring the null DC subcarrier.
    """
    def __init__(self, fft_size, cp_length, num_subcarriers):
        super().__init__()
        if num_subcarriers % 2 != 0:
            raise ValueError("Number of subcarriers must be even.")
        self.fft_size = fft_size
        self.cp_length = cp_length
        self.num_subcarriers = num_subcarriers
        self.symbol_length = fft_size + cp_length

    def forward(self, y_time_serial):
        """
        Args:
            y_time_serial (torch.Tensor): Received time-domain signal.
                                          Shape: [batch_size, num_samples]

        Returns:
            torch.Tensor: Demodulated resource grid.
                          Shape: [batch_size, num_symbols, num_subcarriers]
        """
        batch_size, num_samples = y_time_serial.shape
        num_sym = num_samples // self.symbol_length

        y_time_cp = y_time_serial.reshape(batch_size, num_sym, self.symbol_length)
        y_time = y_time_cp[..., self.cp_length:]
        y_fft = fft.fft(y_time, n=self.fft_size, dim=-1)

        # Extract subcarriers from both sides of the DC bin
        half_sc = self.num_subcarriers // 2
        # Negative frequencies
        neg_freqs = y_fft[..., -half_sc:]
        # Positive frequencies
        pos_freqs = y_fft[..., 1:half_sc + 1]

        return torch.cat([neg_freqs, pos_freqs], dim=-1)

if __name__ == '__main__':
    from ofdm_autoencoder.utils.config import get_config

    print("Verifying OFDM Modulator and Demodulator with DC carrier nulling...")
    
    config = get_config("3")
    N_FFT = config['fft_size']
    N_CP = config['cp_length']
    N_SC = config['num_subcarriers']
    N_SYM = config['symbols_per_slot']
    BATCH_SIZE = 4

    # Ensure num_subcarriers is even for the test
    if N_SC % 2 != 0:
        N_SC -=1
        print(f"Adjusting N_SC to be even for testing: {N_SC}")

    test_grid = torch.randn(BATCH_SIZE, N_SYM, N_SC, dtype=torch.cfloat)
    
    modulator = OFDMModulator(N_FFT, N_CP, N_SC)
    demodulator = OFDMDemodulator(N_FFT, N_CP, N_SC)

    print("\n--- Modulator Test ---")
    time_signal = modulator(test_grid)
    print(f"Input grid shape: {test_grid.shape}")
    print(f"Output time signal shape: {time_signal.shape}")
    expected_len = N_SYM * (N_FFT + N_CP)
    print(f"Expected time signal length: {expected_len}")
    assert time_signal.shape == (BATCH_SIZE, expected_len), "Modulator output shape is incorrect."
    print("Modulator output shape is correct.")

    print("\n--- Demodulator Test ---")
    reconstructed_grid = demodulator(time_signal)
    print(f"Input time signal shape: {time_signal.shape}")
    print(f"Reconstructed grid shape: {reconstructed_grid.shape}")
    assert reconstructed_grid.shape == test_grid.shape, "Demodulator output shape is incorrect."
    print("Demodulator output shape is correct.")

    print("\n--- Loopback Test ---")
    is_close = torch.allclose(test_grid, reconstructed_grid, atol=1e-5)
    mean_error = torch.mean(torch.abs(test_grid - reconstructed_grid))
    print(f"Reconstructed grid is close to original: {is_close}")
    print(f"Mean absolute error: {mean_error.item()}")
    assert is_close, "OFDM Mod/Demod loopback test FAILED."
    print("OFDM Mod/Demod loopback test successful!")