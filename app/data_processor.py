import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
from app.autoencoder_manager import AutoencoderManager
from app.data_handler import load_csv, write_csv
from app.reconstruction import unwindow_data
from app.config_handler import save_debug_info, remote_log
from keras.models import Sequential, Model, load_model

def create_sliding_windows(data, window_size):
    data_array = data.to_numpy()
    dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data_array,
        targets=None,
        sequence_length=window_size,
        sequence_stride=1,
        batch_size=1
    )

    windows = []
    for batch in dataset:
        windows.append(batch.numpy().flatten())

    return pd.DataFrame(windows)

import numpy as np

def process_data(config):
    """
    Process multiple datasets and validation datasets by loading them from CSV files and returning
    windows of shape (sequence_length, num_channels), where:
    - sequence_length: the number of features (columns) in each dataset.
    - num_channels: the number of datasets.
    
    The output is compatible with Conv1D, omitting the batch size.

    Args:
        config (dict): Configuration dictionary with file paths, headers, and other settings.

    Returns:
        tuple:
            - processed_data (dict): A dictionary where each key is a column name, and each value is
              an array of shape (sequence_length, num_channels).
            - validation_data (dict): A dictionary where each key is a column name, and each value is
              an array of shape (sequence_length, num_channels).
    """
    
    # Load the datasets and validation datasets
    print(f"Loading data from CSV files: {config['input_files']}")
    datasets, validation_datasets = load_csv_files(config['input_files'], config['validation_file_paths'], headers=config['headers'])
    
    # Assume all datasets have the same number of rows and columns
    num_datasets = len(datasets)  # This will be the number of channels
    num_columns = datasets[0].shape[1]  # This will be the sequence length

    print(f"Number of datasets (channels): {num_datasets}, Number of features (sequence_length): {num_columns}")

    # Initialize dictionaries to hold processed data and validation data
    processed_data = {}
    validation_data = {}

    # Stack the datasets to create windows with multiple channels (stacked along the last axis)
    for col in range(num_columns):
        # Stack the column values from each dataset for the current column across all datasets
        processed_data[f'col_{col}'] = np.stack([dataset.iloc[:, col].values for dataset in datasets], axis=-1)
    
    # Stack the validation datasets in the same way if they exist
    if validation_datasets:
        print(f"Processing validation datasets with shape: {validation_datasets[0].shape} and number of validation datasets: {len(validation_datasets)}")
        for col in range(num_columns):
            validation_data[f'col_{col}'] = np.stack([validation_dataset.iloc[:, col].values for validation_dataset in validation_datasets], axis=-1)

    return processed_data, validation_data


def run_autoencoder_pipeline(config, encoder_plugin, decoder_plugin):
    start_time = time.time()
    
    print("Running process_data...")
    processed_data,validation_data = process_data(config)
    print("Processed data received.")
    mse=0
    mae=0
    for column, windowed_data in processed_data.items():
        print(f"Processing column: {column}")
        
        # Training loop to optimize the latent space size
        initial_size = config['initial_size']
        step_size = config['step_size']
        threshold_error = config['threshold_error']
        training_batch_size = config['batch_size']
        epochs = config['epochs']
        incremental_search = config['incremental_search']
        
        # Perform unwindowing of the decoded data once
        reconstructed_data = []
        current_size = initial_size
        while True:
            print(f"Training with interface size: {current_size}")
            
            # Create a new instance of AutoencoderManager for each iteration
            autoencoder_manager = AutoencoderManager(encoder_plugin, decoder_plugin)
            
            # Build new autoencoder model with the current size
            autoencoder_manager.build_autoencoder(config['window_size'], current_size, config)

            # Train the autoencoder model
            autoencoder_manager.train_autoencoder(windowed_data, epochs=epochs, batch_size=training_batch_size)

            # Encode and decode the validation data
            encoded_data = autoencoder_manager.encode_data(validation_data[column])  
            decoded_data = autoencoder_manager.decode_data(encoded_data)

            # Check if the decoded data needs reshaping
            if len(decoded_data.shape) == 3:
                decoded_data = decoded_data.reshape(decoded_data.shape[0], decoded_data.shape[1])

            # Perform unwindowing of the decoded data once
            reconstructed_data = unwindow_data(pd.DataFrame(decoded_data))

            # Trim the data to match sizes
            min_size = min(validation_data[column].shape[0], reconstructed_data.shape[0])
            validation_trimmed = validation_data[column][:min_size]
            reconstructed_trimmed = reconstructed_data[:min_size]

            # Ensure both trimmed arrays are NumPy arrays
            validation_trimmed = np.asarray(validation_trimmed)
            reconstructed_trimmed = np.asarray(reconstructed_trimmed)

            # Calculate the MSE and MAE
            mse = autoencoder_manager.calculate_mse(validation_trimmed, reconstructed_trimmed)
            mae = autoencoder_manager.calculate_mae(validation_trimmed, reconstructed_trimmed)
           
            print(f"Mean Squared Error for column {column} with interface size {current_size}: {mse}")
            print(f"Mean Absolute Error for column {column} with interface size {current_size}: {mae}")

            if (incremental_search and mae <= threshold_error) or (not incremental_search and mae >= threshold_error):
                print(f"Optimal interface size found: {current_size} with MSE: {mse} and MAE: {mae}")
                break
            else:
                if incremental_search:
                    current_size += step_size
                else:
                    current_size -= step_size
                if current_size > windowed_data.shape[1] or current_size <= 0:
                    print(f"Cannot adjust interface size beyond data dimensions. Stopping.")
                    break

        encoder_model_filename = f"{config['save_encoder']}_{column}.keras"
        decoder_model_filename = f"{config['save_decoder']}_{column}.keras"
        autoencoder_manager.save_encoder(encoder_model_filename)
        autoencoder_manager.save_decoder(decoder_model_filename)
        print(f"Saved encoder model to {encoder_model_filename}")
        print(f"Saved decoder model to {decoder_model_filename}")

        # save reconstructed data
        output_filename = os.path.splitext(config['output_file'])[0] + f"_{column}.csv"
        write_csv(output_filename, reconstructed_data, include_date=config['force_date'], headers=config['headers'])
        print(f"Output written to {output_filename}")

        print(f"Encoder Dimensions: {autoencoder_manager.encoder_model.input_shape} -> {autoencoder_manager.encoder_model.output_shape}")
        print(f"Decoder Dimensions: {autoencoder_manager.decoder_model.input_shape} -> {autoencoder_manager.decoder_model.output_shape}")

    # Save final configuration and debug information
    end_time = time.time()
    execution_time = end_time - start_time
    debug_info = {
        'execution_time': execution_time,
        'encoder': encoder_plugin.get_debug_info(),
        'decoder': decoder_plugin.get_debug_info(),
        'mse': mse,
        'mae': mae
    }

    # save debug info
    if 'save_log' in config:
        if config['save_log'] != None:
            save_debug_info(debug_info, config['save_log'])
            print(f"Debug info saved to {config['save_log']}.")

    # remote log debug info and config
    if 'remote_log' in config:
        if config['remote_log'] != None:
            remote_log(config, debug_info, config['remote_log'], config['username'], config['password'])
            print(f"Debug info saved to {config['remote_log']}.")

    print(f"Execution time: {execution_time} seconds")

def load_and_evaluate_encoder(config):
    model = load_model(config['load_encoder'])
    print(f"Encoder model loaded from {config['load_encoder']}")
    # Load the input data
    processed_data, validation_data = process_data(config)
    column = list(processed_data.keys())[0]
    windowed_data = processed_data[column]
    # Encode the data
    print(f"Encoding data with shape: {windowed_data.shape}")
    encoded_data = model.predict(windowed_data)
    print(f"Encoded data shape: {encoded_data.shape}")
    # Check if the decoded data needs reshaping
    if len(encoded_data.shape) == 3:
        encoded_data = encoded_data.reshape(encoded_data.shape[0], model.output_shape[2])
    # Save the encoded data to CSV
    evaluate_filename = config['evaluate_encoder']
    np.savetxt(evaluate_filename, encoded_data, delimiter=",")
    print(f"Encoded data saved to {evaluate_filename}")


def load_and_evaluate_decoder(config):
    model = load_model(config['load_decoder'])
    print(f"Decoder model loaded from {config['load_decoder']}")
    # Load the input data
    processed_data = process_data(config)
    column = list(processed_data.keys())[0]
    windowed_data = processed_data[column]
    # Decode the data
    print(f"Decoding data with shape: {windowed_data.shape}")
    decoded_data = model.predict(windowed_data)
    print(f"Decoded data shape: {decoded_data.shape}")
    # Check if the decoded data needs reshaping
    if len(decoded_data.shape) == 3:
        decoded_data = decoded_data.reshape(decoded_data.shape[0], decoded_data.shape[2])
    # Save the encoded data to CSV
    evaluate_filename = config['evaluate_decoder']
    np.savetxt(evaluate_filename, decoded_data, delimiter=",")
    print(f"Decoded data saved to {evaluate_filename}")
