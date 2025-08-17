"""
Implementation of the CNN Encoder (Transmitter) as described in
Section 3.2 and Table 2 of the Readme.md.
"""
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    A convolutional encoder that maps a block of K bits to a
    complex-valued OFDM resource grid of size [N_sc x N_sym].
    The architecture uses transposed convolutions to upsample
    an embedded representation of the input bits.
    """
    def __init__(self, k_bits, num_subcarriers, num_symbols):
        super().__init__()
        self.k_bits = k_bits
        self.num_subcarriers = num_subcarriers
        self.num_symbols = num_symbols

        # Calculate an intermediate dense layer size.
        # This is a hyperparameter and can be tuned. A simple starting point:
        dense_size = 1024

        self.network = nn.Sequential(
            # 1. Input Embedding
            nn.Linear(k_bits, dense_size),
            nn.ReLU(inplace=True),

            # 2. Reshape for Convolution. The shape will be [batch, channels, height, width]
            # Height and width start at 1x1.
            nn.Unflatten(1, (dense_size, 1, 1)),

            # 3. Deterministic Convolutional Upsampling stack.
            # This architecture is designed to precisely output the target grid size
            # without requiring the problematic AdaptiveAvgPool2d layer.
            # This example is tailored for the 3MHz (200x6) case.
            
            # Upsample from (dense_size, 1, 1)
            nn.ConvTranspose2d(dense_size, 512, kernel_size=(25, 3), stride=(1, 1), padding=0), # -> (512, 25, 3)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Upsample height by 4x, width by 2x
            # H_out = (25-1)*4 + 4 = 100
            # W_out = (3-1)*2 + 2 = 6
            nn.ConvTranspose2d(512, 256, kernel_size=(4, 2), stride=(4, 2), padding=0), # -> (256, 100, 6)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Final upsample to target height
            # H_out = (100-1)*2 + 2 = 200
            # W_out = (6-1)*1 + 1 = 6
            nn.ConvTranspose2d(256, 128, kernel_size=(2, 1), stride=(2, 1), padding=0), # -> (128, 200, 6)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 4. Output Layer
            # Project to 2 channels (I and Q) without changing spatial dimensions.
            nn.Conv2d(128, 2, kernel_size=(1, 1)) # -> (2, 200, 6)
        )

    def forward(self, b):
        """
        Args:
            b (torch.Tensor): Input bit tensor.
                              Shape: [batch_size, k_bits]
        
        Returns:
            torch.Tensor: Real-valued resource grid representing I/Q components.
                          Shape: [batch_size, N_sc, N_sym, 2]
        """
        # The network outputs [batch, 2, N_sc, N_sym]
        x = self.network(b) 
        
        # Permute to get [batch, N_sc, N_sym, 2] for compatibility with downstream modules
        # Permute and ensure tensor is contiguous in memory
        return x.permute(0, 2, 3, 1).contiguous()

if __name__ == '__main__':
    from ofdm_autoencoder.utils.config import get_config

    print("Verifying Encoder architecture...")
    
    # Use 3 MHz config for testing
    config = get_config("3")
    K = config['k_bits']
    N_SC = config['num_subcarriers']
    N_SYM = config['symbols_per_slot']
    BATCH_SIZE = 4
    
    # Create dummy input
    # Input to the encoder is typically floats (0.0 or 1.0) or after an embedding.
    test_bits = torch.randint(0, 2, (BATCH_SIZE, K)).float()

    # Instantiate the encoder
    encoder = Encoder(k_bits=K, num_subcarriers=N_SC, num_symbols=N_SYM)
    
    print("\n--- Encoder Test ---")
    print(f"Input bit tensor shape: {test_bits.shape}")
    
    # Get output
    output_grid = encoder(test_bits)
    
    print(f"Output grid shape: {output_grid.shape}")
    
    expected_shape = (BATCH_SIZE, N_SC, N_SYM, 2)
    print(f"Expected output shape: {expected_shape}")
    
    assert output_grid.shape == expected_shape, "Encoder output shape is incorrect."
    
    print("Encoder output shape is correct.")
    print("Encoder verification successful!")