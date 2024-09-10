import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Feature-fusion: A tool for merging datasets and reducing dimensionality with autoencoders, supporting dynamic plugins.")
    
    # Updated to input_files
    parser.add_argument('--input_files', nargs='+', type=str, help='Paths to the input CSV files to be merged.')

    parser.add_argument('--validation_files', type=str, help='Path to the input CSV files used to test the trained autoencoder.')
    
    # Retained optional arguments
    parser.add_argument('--output_file', type=str, help='Path to the output CSV file.')
    parser.add_argument('--save_encoder', type=str, help='Filename to save the trained encoder model.')
    parser.add_argument('--save_decoder', type=str, help='Filename to save the trained decoder model.')
    parser.add_argument('--load_encoder', type=str, help='Filename to load encoder parameters from.')
    parser.add_argument('--load_decoder', type=str, help='Filename to load decoder parameters from.')
    parser.add_argument('--evaluate_encoder', type=str, help='Filename for outputting encoder evaluation results.')
    parser.add_argument('--evaluate_decoder', type=str, help='Filename for outputting decoder evaluation results.')
    
    # Plugins
    parser.add_argument('--encoder_plugin', type=str, help='Name of the encoder plugin to use (e.g., CNN, ANN, LSTM, etc.).')
    parser.add_argument('--decoder_plugin', type=str, help='Name of the decoder plugin to use (e.g., CNN, ANN, LSTM, etc.).')
    
    # Time series processing
    parser.add_argument('--window_size', type=int, help='Sliding window size to use for processing time series data.')
    
    # Training settings
    parser.add_argument('--threshold_error', type=float, help='MSE error threshold to stop the training process.')
    parser.add_argument('--initial_size', type=int, help='Initial size of the encoder/decoder interface (latent space size).')
    parser.add_argument('--step_size', type=int, help='Step size to adjust the encoder/decoder interface during incremental or decremental search.')
    
    # Remote logging and configurations
    parser.add_argument('--remote_log', type=str, help='URL of a remote API endpoint for saving debug variables in JSON format.')
    parser.add_argument('--remote_load_config', type=str, help='URL of a remote JSON configuration file to download and execute.')
    parser.add_argument('--remote_save_config', type=str, help='URL of a remote API endpoint for saving the configuration in JSON format.')
    
    # Authentication
    parser.add_argument('--username', type=str, help='Username for the API endpoint.')
    parser.add_argument('--password', type=str, help='Password for the API endpoint.')
    
    # Configuration and logging
    parser.add_argument('--load_config', type=str, help='Path to load a configuration file from disk.')
    parser.add_argument('--save_config', type=str, help='Path to save the current configuration.')
    parser.add_argument('--save_log', type=str, help='Path to save the current debug info (logs, intermediate results).')

    # Miscellaneous
    parser.add_argument('--quiet_mode', action='store_true', help='Suppress output messages and minimize terminal logging during execution.')
    parser.add_argument('--force_date', action='store_true', help='Force inclusion of the date in the output CSV files.')
    parser.add_argument('--incremental_search', action='store_true', help='Enable incremental search for adjusting the encoder/decoder interface (latent space).')
    parser.add_argument('--headers', action='store_true', help='Indicate if the CSV input files contain headers.')

    return parser.parse_known_args()
