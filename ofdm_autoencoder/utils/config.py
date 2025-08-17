"""
Centralized configuration for system and OFDM parameters.
This file holds the parameters for different bandwidth settings as specified
in the project's Readme.md (Table 1).
"""

import math

# Common parameters across all configurations
SUBCARRIER_SPACING_HZ = 15_000  # 15 kHz
SYMBOLS_PER_SLOT = 6
FRAME_DURATION_S = 0.01  # 10 ms
SYMBOLS_PER_FRAME = 120
SLOTS_PER_FRAME = SYMBOLS_PER_FRAME // SYMBOLS_PER_SLOT
SLOT_DURATION_S = FRAME_DURATION_S / SLOTS_PER_FRAME

# Bandwidth-specific configurations
PARAMS_3MHZ = {
    "bandwidth_mhz": "3",
    "fft_size": 256,
    "num_subcarriers": 200,
    "cp_length": 18,  # Scaled Normal CP
    "k_bits": 48, # TBD, placeholder
}

PARAMS_20MHZ = {
    "bandwidth_mhz": "20",
    "fft_size": 2048,
    "num_subcarriers": 1200,
    "cp_length": 144,  # Normal CP for 2048 FFT
    "k_bits": 320,  # TBD, placeholder
}

def get_config(bandwidth_mhz="3"):
    """
    Returns the parameter dictionary for a given bandwidth.
    """
    if str(bandwidth_mhz) == "3":
        config = PARAMS_3MHZ
    elif str(bandwidth_mhz) == "20":
        config = PARAMS_20MHZ
    else:
        raise ValueError(f"Unsupported bandwidth: {bandwidth_mhz} MHz. Choose '3' or '20'.")

    # Add derived and common parameters
    config["subcarrier_spacing_hz"] = SUBCARRIER_SPACING_HZ
    config["symbols_per_slot"] = SYMBOLS_PER_SLOT
    config["useful_symbol_time_s"] = 1 / SUBCARRIER_SPACING_HZ
    config["sampling_freq_hz"] = config["fft_size"] * SUBCARRIER_SPACING_HZ
    config["cp_time_s"] = config["cp_length"] / config["sampling_freq_hz"]
    config["symbol_duration_s"] = config["useful_symbol_time_s"] + config["cp_time_s"]
    config["encoder_output_size"] = (config["num_subcarriers"], SYMBOLS_PER_SLOT)
    config["decoder_input_size"] = (config["num_subcarriers"], SYMBOLS_PER_SLOT)
    config["decoder_output_size"] = config["k_bits"]

    # Calculate Eb/No to SNR conversion factor
    # SNR = (Eb/No) * (Bitrate / Bandwidth)
    # Bandwidth = N_sc * Subcarrier_Spacing
    # Bitrate = K_bits / (N_sym * Symbol_Duration)
    # Factor = K_bits / (N_sym * Symbol_Duration * N_sc * Subcarrier_Spacing)
    bitrate = config["k_bits"] / (SYMBOLS_PER_SLOT * config["symbol_duration_s"])
    bandwidth = config["num_subcarriers"] * SUBCARRIER_SPACING_HZ
    config["ebno_to_snr_factor"] = bitrate / bandwidth

    return config

if __name__ == '__main__':
    # Print out the configurations for verification
    print("--- 3 MHz Configuration ---")
    config_3mhz = get_config("3")
    for key, val in config_3mhz.items():
        print(f"{key:<25}: {val}")

    print("\n--- 20 MHz Configuration ---")
    config_20mhz = get_config("20")
    for key, val in config_20mhz.items():
        print(f"{key:<25}: {val}")