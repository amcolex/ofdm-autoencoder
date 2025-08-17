"""
Implementation of the CNN Decoder (Receiver) as described in
Section 3.3 and Table 3 of the Readme.md.
"""
import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    A convolutional decoder that maps a distorted, complex-valued OFDM
    resource grid of size [N_sc x N_sym] back to a block of K bits.
    The architecture uses standard convolutions to downsample the grid and
    a dense classification head to recover the bits.
    """
    def __init__(self, k_bits, num_subcarriers, num_symbols):
        super().__init__()
        self.k_bits = k_bits
        self.num_subcarriers = num_subcarriers
        self.num_symbols = num_symbols

        # This architecture is based on Table 3 and is designed to be
        # roughly symmetrical to the encoder.
        self.network = nn.Sequential(
            # Input shape: [batch, 2, N_sc, N_sym]
            
            # Conv Layer 1
            nn.Conv2d(2, 64, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Conv Layer 2
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Conv Layer 3
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Conv Layer 4
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Flatten the output of the conv stack
            nn.Flatten(),
        )

        # A dummy forward pass is needed to determine the flattened size
        # for the subsequent dense layers.
        with torch.no_grad():
            dummy_input = torch.zeros(1, 2, self.num_subcarriers, self.num_symbols)
            dummy_output = self.network(dummy_input)
            flattened_size = dummy_output.shape[1]

        self.dense_head = nn.Sequential(
            # Dense Classification Head
            nn.Linear(flattened_size, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            
            # Output layer
            nn.Linear(4096, self.k_bits),
            nn.Sigmoid() # Sigmoid activation for bit probabilities
        )

    def forward(self, y_grid_real):
        """
        Args:
            y_grid_real (torch.Tensor): The received grid, split into real and
                                        imaginary parts.
                                        Shape: [batch, N_sc, N_sym, 2]
        
        Returns:
            torch.Tensor: A tensor of probabilities for each of the K bits.
                          Shape: [batch, K]
        """
        # Permute input to [batch, 2, N_sc, N_sym] for Conv2D layers
        # Permute input and ensure it's contiguous before the conv network
        y = y_grid_real.permute(0, 3, 1, 2).contiguous()
        
        features = self.network(y)
        b_hat = self.dense_head(features)
        
        return b_hat

if __name__ == '__main__':
    from ofdm_autoencoder.utils.config import get_config

    print("Verifying Decoder architecture...")
    
    # Use 3 MHz config for testing
    config = get_config("3")
    K = config['k_bits']
    N_SC = config['num_subcarriers']
    N_SYM = config['symbols_per_slot']
    BATCH_SIZE = 4
    
    # Create dummy input
    test_grid = torch.randn(BATCH_SIZE, N_SC, N_SYM, 2)

    # Instantiate the decoder
    decoder = Decoder(k_bits=K, num_subcarriers=N_SC, num_symbols=N_SYM)
    
    print("\n--- Decoder Test ---")
    print(f"Input grid tensor shape: {test_grid.shape}")
    
    # Get output
    output_bits = decoder(test_grid)
    
    print(f"Output bit probability shape: {output_bits.shape}")
    
    expected_shape = (BATCH_SIZE, K)
    print(f"Expected output shape: {expected_shape}")
    
    assert output_bits.shape == expected_shape, "Decoder output shape is incorrect."
    
    # Check that output values are in the range [0, 1]
    assert torch.all(output_bits >= 0) and torch.all(output_bits <= 1), \
        "Output probabilities are not in the [0, 1] range."
        
    print("Decoder output shape and value range are correct.")
    print("Decoder verification successful!")