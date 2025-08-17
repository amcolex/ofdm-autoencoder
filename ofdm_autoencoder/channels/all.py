"""
Differentiable channel models as described in Sections 2.4 and 5.3-5.4 of the Readme.md.
Includes implementations for a Tapped Delay Line (TDL) Rayleigh Fading channel,
a residual Carrier Frequency Offset (CFO) impairment, and Additive White Gaussian Noise (AWGN).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TDLChannel(nn.Module):
    """
    Simulates a time-varying, frequency-selective Tapped Delay Line (TDL)
    Rayleigh fading channel using a differentiable grouped 1D convolution.
    This allows a different channel impulse response per item in the batch.
    """
    def __init__(self, num_taps, apply_prob=1.0):
        super().__init__()
        self.num_taps = num_taps
        self.apply_prob = apply_prob
        # Manual padding to ensure output length equals input length
        self.padding = (self.num_taps - 1) // 2

    def forward(self, x_time):
        batch_size, num_samples = x_time.shape

        if not self.training or torch.rand(1).item() > self.apply_prob:
            return x_time

        # Generate random complex Gaussian taps for each batch entry
        # Shape must be (C_out=batch_size, C_in/groups=1, K_size=num_taps)
        taps_real = torch.randn(batch_size, 1, self.num_taps, device=x_time.device)
        taps_imag = torch.randn(batch_size, 1, self.num_taps, device=x_time.device)

        # Reshape for grouped convolution: input shape (N=1, C=batch_size, L=num_samples)
        x_time_reshaped = x_time.unsqueeze(0)
        
        # Split into real/imag parts
        x_real, x_imag = x_time_reshaped.real, x_time_reshaped.imag
        
        # Apply channel via complex convolution implemented with two real grouped convolutions
        # Each item in the batch is its own group
        y_real = F.conv1d(x_real, taps_real, padding=self.padding, groups=batch_size) - F.conv1d(x_imag, taps_imag, padding=self.padding, groups=batch_size)
        y_imag = F.conv1d(x_real, taps_imag, padding=self.padding, groups=batch_size) + F.conv1d(x_imag, taps_real, padding=self.padding, groups=batch_size)

        # Combine back to complex and reshape to [batch_size, num_samples]
        y_time_complex_reshaped = torch.complex(y_real, y_imag)
        return y_time_complex_reshaped.squeeze(0)

class ResidualCFO(nn.Module):
    """
    Applies a random residual Carrier Frequency Offset (CFO) to the time-domain signal.
    """
    def __init__(self, max_cfo_norm, subcarrier_spacing, sampling_freq, apply_prob=1.0):
        super().__init__()
        self.max_cfo_norm = max_cfo_norm # Max CFO normalized by subcarrier spacing
        self.subcarrier_spacing = subcarrier_spacing
        self.sampling_freq = sampling_freq
        self.apply_prob = apply_prob

    def forward(self, x_time):
        if not self.training or torch.rand(1).item() > self.apply_prob:
            return x_time

        batch_size, num_samples = x_time.shape
        
        cfo_norm = (torch.rand(batch_size, 1, device=x_time.device) * 2 - 1) * self.max_cfo_norm
        delta_f = cfo_norm * self.subcarrier_spacing
        
        t = torch.arange(0, num_samples, device=x_time.device) / self.sampling_freq
        phase_rotation = 2.0 * torch.pi * delta_f * t
        
        cfo_phasor = torch.exp(1j * phase_rotation)
        return x_time * cfo_phasor

class AWGN(nn.Module):
    """
    Adds Additive White Gaussian Noise to the signal based on a target Eb/No.
    """
    def __init__(self, ebno_to_snr_factor):
        super().__init__()
        self.ebno_to_snr_factor = ebno_to_snr_factor

    def forward(self, x_time, ebno_db):
        # Convert Eb/No to SNR
        # SNR = (Eb/No) * (Bitrate / Bandwidth)
        snr_db = ebno_db + 10 * torch.log10(torch.tensor(self.ebno_to_snr_factor, device=x_time.device))
        
        # Calculate noise standard deviation from SNR
        # Signal power is normalized to 1, so SNR = 1 / Noise_Power
        noise_power_db = -snr_db
        noise_std_dev = (10**(noise_power_db / 10.0)).sqrt()

        # Generate complex noise
        # The std dev is split between the real and imaginary components
        noise = torch.randn_like(x_time) * (noise_std_dev.view(-1, 1) / (2**0.5))
        return x_time + noise

class WirelessChannel(nn.Module):
    """
    A composite channel model that combines CFO, Fading, and AWGN.
    This module is used to enable/disable impairments for curriculum learning.
    """
    def __init__(self, config):
        super().__init__()
        self.use_fading = False
        self.use_cfo = False

        self.fading_channel = TDLChannel(num_taps=5)
        self.cfo_channel = ResidualCFO(
            max_cfo_norm=0.05, # Max offset is 5% of subcarrier spacing
            subcarrier_spacing=config['subcarrier_spacing_hz'],
            sampling_freq=config['sampling_freq_hz']
        )
        self.awgn_channel = AWGN(config['ebno_to_snr_factor'])

    def forward(self, x_time, ebno_db):
        if self.use_cfo:
            x_time = self.cfo_channel(x_time)
        if self.use_fading:
            x_time = self.fading_channel(x_time)
        
        y_time = self.awgn_channel(x_time, ebno_db)
        return y_time

    def set_fading(self, enabled: bool):
        print(f"Wireless Channel: Fading set to {enabled}")
        self.use_fading = enabled

    def set_cfo(self, enabled: bool):
        print(f"Wireless Channel: CFO set to {enabled}")
        self.use_cfo = enabled

if __name__ == '__main__':
    from ofdm_autoencoder.utils.config import get_config

    print("Verifying channel models...")
    config = get_config("3")
    BATCH_SIZE = 4
    N_SAMPLES = (config['fft_size'] + config['cp_length']) * config['symbols_per_slot']
    
    test_signal = torch.randn(BATCH_SIZE, N_SAMPLES, dtype=torch.cfloat)
    test_ebno = torch.tensor([10.0] * BATCH_SIZE)

    # --- Test Composite Channel ---
    print("\n--- Composite WirelessChannel Test ---")
    channel = WirelessChannel(config)
    
    print("\n1. AWGN only")
    channel.set_fading(False)
    channel.set_cfo(False)
    output = channel(test_signal, test_ebno)
    assert output.shape == test_signal.shape
    print("AWGN-only output shape is correct.")

    print("\n2. AWGN + Fading")
    channel.set_fading(True)
    channel.set_cfo(False)
    output = channel(test_signal, test_ebno)
    assert output.shape == test_signal.shape
    print("AWGN+Fading output shape is correct.")
    
    print("\n3. AWGN + Fading + CFO")
    channel.set_fading(True)
    channel.set_cfo(True)
    output = channel(test_signal, test_ebno)
    assert output.shape == test_signal.shape
    print("AWGN+Fading+CFO output shape is correct.")
    
    print("\nChannel model verification successful!")