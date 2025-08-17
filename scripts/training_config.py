"""
Configuration for the training script.
This file centralizes all hyperparameters and settings for a training run.
"""
from datetime import datetime

# --- Run Settings ---
RUN_NAME = f"3MHz_Phase1_AWGN_{datetime.now():%Y%m%d_%H%M%S}"
BANDWIDTH_MHZ = "3"  # System bandwidth ("3" or "20")
LOAD_CHECKPOINT = None # Path to a model checkpoint to resume training, e.g., "runs/your_run_name/best_model.pth"

# --- Training Hyperparameters ---
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-4

# --- Loop Settings ---
STEPS_PER_EPOCH = 500  # Number of training batches per epoch
VAL_STEPS = 100        # Number of validation batches per epoch

# --- Curriculum Learning Phases ---
# Configure which impairments are active for this training run.
# The training script will print the active configuration.
ENABLE_FADING = False
ENABLE_CFO = False

# --- Eb/No Settings ---
# For training, Eb/No can be a fixed value or a range for the model to generalize.
# For a fixed value (e.g., Phase 1), set min and max to the same value.
EBNO_DB_TRAIN_MIN = 15.0  # Minimum Eb/No in dB for training
EBNO_DB_TRAIN_MAX = 15.0  # Maximum Eb/No in dB for training

# For validation, a fixed Eb/No is used to have a consistent benchmark.
EBNO_DB_EVAL = 15.0