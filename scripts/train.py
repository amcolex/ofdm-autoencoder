"""
Main training script for the OFDM Autoencoder.

This script implements the phased training plan (curriculum learning)
by importing settings from a dedicated configuration file.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import matplotlib.pyplot as plt

# Adjust path to import from the project's source directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import model components and utilities
from ofdm_autoencoder.utils.config import get_config
from ofdm_autoencoder.models.encoder import Encoder
from ofdm_autoencoder.models.decoder import Decoder
from ofdm_autoencoder.models.ofdm import OFDMModulator, OFDMDemodulator
from ofdm_autoencoder.models.autoencoder import OFDMAutoencoder
from ofdm_autoencoder.channels.all import WirelessChannel

# Import training settings
import training_config as TRAIN_CFG

def calculate_ber(y_true, y_pred_probs):
    """Calculate Bit Error Rate."""
    y_pred = (y_pred_probs > 0.5).float()
    return (y_true != y_pred).float().mean()

def plot_constellation(points, epoch, save_dir):
    """Plots the I/Q constellation points and saves the figure."""
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], s=12, alpha=0.7)
    plt.title(f'Learned Constellation - Epoch {epoch}', fontsize=16)
    plt.xlabel("I-Component", fontsize=12)
    plt.ylabel("Q-Component", fontsize=12)
    plt.grid(True)
    ax = plt.gca()
    lim = np.max(np.abs(points)) * 1.2 + 1e-3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', adjustable='box')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"constellation_epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close()
    return save_path

def main():
    """Main training and evaluation loop."""
    # --- Setup ---
    # Determine the correct device to use
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load system and training configurations
    sys_config = get_config(TRAIN_CFG.BANDWIDTH_MHZ)
    k_bits = sys_config['k_bits']
    
    # Setup TensorBoard
    log_dir = os.path.join("runs", TRAIN_CFG.RUN_NAME)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    # --- Model Initialization ---
    encoder = Encoder(k_bits, sys_config['num_subcarriers'], sys_config['symbols_per_slot'])
    decoder = Decoder(k_bits, sys_config['num_subcarriers'], sys_config['symbols_per_slot'])
    modulator = OFDMModulator(sys_config['fft_size'], sys_config['cp_length'], sys_config['num_subcarriers'])
    demodulator = OFDMDemodulator(sys_config['fft_size'], sys_config['cp_length'], sys_config['num_subcarriers'])
    channel = WirelessChannel(sys_config)
    
    autoencoder = OFDMAutoencoder(encoder, decoder, modulator, demodulator, channel).to(device)
    
    # --- Curriculum Learning Setup ---
    channel.set_fading(TRAIN_CFG.ENABLE_FADING)
    channel.set_cfo(TRAIN_CFG.ENABLE_CFO)
    
    # --- Optimizer and Loss Function ---
    optimizer = optim.Adam(autoencoder.parameters(), lr=TRAIN_CFG.LEARNING_RATE)
    criterion = nn.BCELoss()
    
    # --- Checkpoint Loading (Optional) ---
    if TRAIN_CFG.LOAD_CHECKPOINT:
        checkpoint = torch.load(TRAIN_CFG.LOAD_CHECKPOINT, map_location=device)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {TRAIN_CFG.LOAD_CHECKPOINT}")

    # Create a fixed reference bit sequence for constellation visualization
    # We generate 2^n test points up to a max of 4096 for visual clarity
    num_ref_points = min(2**k_bits, 4096)
    ref_bits = torch.randint(0, 2, (num_ref_points, k_bits), device=device).float()

    best_val_ber = 1.0

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(TRAIN_CFG.EPOCHS):
        autoencoder.train()
        total_train_loss = 0
        
        for step in range(TRAIN_CFG.STEPS_PER_EPOCH):
            optimizer.zero_grad()
            
            bits_in = torch.randint(0, 2, (TRAIN_CFG.BATCH_SIZE, k_bits), device=device).float()
            ebno_db = torch.empty(TRAIN_CFG.BATCH_SIZE, device=device).uniform_(
                TRAIN_CFG.EBNO_DB_TRAIN_MIN, TRAIN_CFG.EBNO_DB_TRAIN_MAX
            )
            
            bits_out = autoencoder(bits_in, ebno_db)
            
            loss = criterion(bits_out, bits_in)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / TRAIN_CFG.STEPS_PER_EPOCH
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        
        # --- Validation Loop ---
        autoencoder.eval()
        total_val_ber = 0
        with torch.no_grad():
            for _ in range(TRAIN_CFG.VAL_STEPS):
                bits_in_val = torch.randint(0, 2, (TRAIN_CFG.BATCH_SIZE, k_bits), device=device).float()
                ebno_db_val = torch.full((TRAIN_CFG.BATCH_SIZE,), TRAIN_CFG.EBNO_DB_EVAL, device=device)
                
                bits_out_val = autoencoder(bits_in_val, ebno_db_val)
                ber = calculate_ber(bits_in_val, bits_out_val)
                total_val_ber += ber.item()
        
        avg_val_ber = total_val_ber / TRAIN_CFG.VAL_STEPS
        writer.add_scalar("BER/validation", avg_val_ber, epoch)
        
        print(f"Epoch {epoch+1}/{TRAIN_CFG.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val BER: {avg_val_ber:.6f} @ {TRAIN_CFG.EBNO_DB_EVAL} dB")
        
        # --- Constellation Visualization ---
        with torch.no_grad():
            constellation_grid_real = autoencoder.encoder(ref_bits)
            constellation_grid_complex = torch.complex(constellation_grid_real[..., 0], constellation_grid_real[..., 1])
            pwr = torch.mean(torch.abs(constellation_grid_complex)**2)
            constellation_normalized = constellation_grid_complex / torch.sqrt(pwr)
            
            constellation_points = constellation_normalized.reshape(-1, 1).cpu().numpy()
            img_path = plot_constellation(np.hstack([constellation_points.real, constellation_points.imag]), epoch + 1, log_dir)
            
            img = plt.imread(img_path)
            writer.add_image('Constellation', img, epoch + 1, dataformats='HWC')

        # --- Model Checkpointing ---
        if avg_val_ber < best_val_ber:
            best_val_ber = avg_val_ber
            checkpoint_path = os.path.join(log_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': autoencoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'ber': avg_val_ber,
            }, checkpoint_path)
            print(f"New best model saved to {checkpoint_path}")

    writer.close()
    print("Training complete.")

if __name__ == '__main__':
    main()