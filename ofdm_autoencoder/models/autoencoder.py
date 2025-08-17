"""
Implementation of the main OFDMAutoencoder model, which assembles the
encoder, decoder, modulator, and demodulator components.
This is based on the description in Section 5.2 of the Readme.md.
"""
import torch
import torch.nn as nn

class OFDMAutoencoder(nn.Module):
    """
    The complete end-to-end OFDM Autoencoder model.
    It encapsulates the encoder, decoder, and the fixed OFDM modulation
    and demodulation blocks. It also handles the power normalization of
    the transmitted signal and the interface with the channel model.
    """
    def __init__(self, encoder, decoder, modulator, demodulator, channel):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.modulator = modulator
        self.demodulator = demodulator
        self.channel = channel

    def forward(self, b, ebno_db):
        """
        The full end-to-end signal chain.

        Args:
            b (torch.Tensor): Input bit tensor.
                              Shape: [batch_size, k_bits]
            ebno_db (torch.Tensor): The desired Eb/No in dB for channel simulation.
                                    Can be a scalar or a tensor of shape [batch_size].

        Returns:
            torch.Tensor: The final output of the decoder, a tensor of bit
                          probabilities. Shape: [batch_size, k_bits]
        """
        # 1. Transmitter Side
        # Encoder maps bits to a real-valued grid [batch, N_sc, N_sym, 2]
        x_grid_real = self.encoder(b)

        # Convert to complex-valued grid for modulation [batch, N_sym, N_sc]
        x_grid_complex = torch.complex(
            x_grid_real[..., 0], x_grid_real[..., 1]
        ).permute(0, 2, 1).contiguous() # Permute to [batch, N_sym, N_sc] for modulator

        # 2. Power Normalization
        # Enforce average power constraint of 1 per complex symbol
        pwr = torch.mean(torch.abs(x_grid_complex)**2, dim=(-1, -2), keepdim=True)
        x_normalized = x_grid_complex / torch.sqrt(pwr)

        # 3. OFDM Modulation
        # Modulator expects [batch, N_sym, N_sc] and outputs [batch, num_samples]
        x_time = self.modulator(x_normalized)

        # 4. Channel
        # The channel model takes the time-domain signal and Eb/No
        y_time = self.channel(x_time, ebno_db)

        # 5. Receiver Side
        # Demodulator expects [batch, num_samples] and outputs [batch, N_sym, N_sc]
        y_grid_complex = self.demodulator(y_time)

        # Permute to [batch, N_sc, N_sym] to match decoder's expectation after split
        y_grid_complex = y_grid_complex.permute(0, 2, 1).contiguous()

        # Convert back to real-valued grid for the decoder [batch, N_sc, N_sym, 2]
        y_grid_real = torch.stack([y_grid_complex.real, y_grid_complex.imag], dim=-1)

        # 6. Decoder
        # Decoder maps the received grid back to bit probabilities
        b_hat = self.decoder(y_grid_real)

        return b_hat

if __name__ == '__main__':
    # This verification requires a placeholder for the channel model
    from ofdm_autoencoder.utils.config import get_config
    from ofdm_autoencoder.models.encoder import Encoder
    from ofdm_autoencoder.models.decoder import Decoder
    from ofdm_autoencoder.models.ofdm import OFDMModulator, OFDMDemodulator

    class DummyChannel(nn.Module):
        """A simple AWGN channel for verification purposes."""
        def __init__(self, ebno_to_snr_factor):
            super().__init__()
            self.ebno_to_snr_factor = ebno_to_snr_factor

        def forward(self, x_time, ebno_db):
            snr_db = ebno_db + 10 * torch.log10(torch.tensor(self.ebno_to_snr_factor))
            noise_power_db = -snr_db
            noise_std_dev = (10**(noise_power_db / 10.0)).sqrt()
            
            # For complex noise, std dev is split between real and imag
            noise = torch.randn_like(x_time) * (noise_std_dev / (2**0.5))
            return x_time + noise

    print("Verifying OFDMAutoencoder model...")

    config = get_config("3")
    K = config['k_bits']
    N_SC = config['num_subcarriers']
    N_SYM = config['symbols_per_slot']
    N_FFT = config['fft_size']
    N_CP = config['cp_length']
    BATCH_SIZE = 4

    # Instantiate all components
    encoder = Encoder(K, N_SC, N_SYM)
    decoder = Decoder(K, N_SC, N_SYM)
    modulator = OFDMModulator(N_FFT, N_CP, N_SC)
    demodulator = OFDMDemodulator(N_FFT, N_CP, N_SC)
    channel = DummyChannel(config['ebno_to_snr_factor'])

    # Assemble the autoencoder
    autoencoder = OFDMAutoencoder(encoder, decoder, modulator, demodulator, channel)

    # Create dummy input data
    test_bits = torch.randint(0, 2, (BATCH_SIZE, K)).float()
    test_ebno = torch.tensor([10.0] * BATCH_SIZE) # 10 dB

    print("\n--- Autoencoder Test ---")
    print(f"Input bits shape: {test_bits.shape}")
    
    # Run a forward pass
    output_bits = autoencoder(test_bits, test_ebno)
    
    print(f"Output bit probabilities shape: {output_bits.shape}")
    
    expected_shape = (BATCH_SIZE, K)
    print(f"Expected output shape: {expected_shape}")
    
    assert output_bits.shape == expected_shape, "Autoencoder output shape is incorrect."
    
    print("Autoencoder forward pass successful and output shape is correct.")