
# Feature Fusion 

## Description

The Feature-fusion is a powerful and flexible tool designed for merging datasets from multiple feature-extractors and training autoencoders on the concatenated data. It allows for reducing the dimensionality of the merged datasets or achieving a specified reconstruction error threshold dynamically by adjusting the size of the latent space. Like feature-extractor, it supports dynamic plugins and configurable parameters, making it adaptable for a wide range of data types.

### Key Features:
- **Dynamic Plugins:** Easily switch between different encoder and decoder plugins (e.g. ANN, CNN, LSTM, Transformer) to find the best fit for your concatenated data.
- **Configurable Parameters:** Customize the fusion and training process with parameters such as the number of output features, initial size, step size, epochs, batch size, and error thresholds.
- **Model Management:** Save and load encoder and decoder models for reuse, avoiding the need to retrain models from scratch after each fusion.
- **Remote Configuration:** Load and save configurations remotely, facilitating seamless integration with other systems and automation pipelines.
- **Incremental Search:** Optimize the latent space size dynamically during training to achieve the best performance based on a specified error threshold.

This tool is designed for data scientists and machine learning engineers who need to preprocess, merge, and encode large datasets efficiently, and it can be easily integrated into larger machine learning workflows.


## Installation Instructions

To install and set up the feature-fusion application, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/harveybc/feature-fusion.git
    cd feature-fusion
    ```

2. **Create and Activate a Virtual Environment (Anaconda is required)**:

    - **Using `conda`**:
        ```bash
        conda create --name feature-fusion-env python=3.9
        conda activate feature-fusion-env
        ```

3. **Install Dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. **Build the Package**:
    ```bash
    python -m build
    ```

5. **Install the Package**:
    ```bash
    pip install .
    ```

6. **(Optional) Run the feature-fusion**:
    - On Windows, run the following command to verify installation (it uses all default values, use feature-fusion.bat --help, for complete command line arguments description):
        ```bash
        feature-fusion.bat tests\data\merged_feature_data.csv 
        ```

    - On Linux, run:
        ```bash
        sh feature-fusion.sh tests\data\merged_feature_data.csv
        ```

7. **(Optional) Run Tests**:
For passing remote tests, requires an instance of [harveybc/data-logger](https://github.com/harveybc/data-logger)
    - On Windows, run the following command to run the tests:
        ```bash
        set_env.bat
        pytest
        ```

    - On Linux, run:
        ```bash
        sh ./set_env.sh
        pytest
        ```

8. **(Optional) Generate Documentation**:
    - Run the following command to generate code documentation in HTML format in the docs directory:
        ```bash
        pdoc --html -o docs app
        ```

9. **(Optional) Install Nvidia CUDA GPU support**:

Please read: [Readme - CUDA](https://github.com/harveybc/feature-fusion/blob/master/README_CUDA.md)

## Usage

The application supports several command line arguments to control its behavior:

```
usage: feature-fusion.bat tests\data\merged_feature_data.csv
```

### Command Line Arguments

#### Required Arguments

- `--input_files` (list[str]): Paths to the input CSV files from different feature-extractors to be merged. This argument allows multiple input files to be concatenated horizontally based on their columns.

#### Optional Arguments

- `--output_file` (str): Path to the output CSV file after fusion and dimensionality reduction.
- `--save_encoder` (str): Filename to save the trained encoder model after the autoencoder training is complete.
- `--save_decoder` (str): Filename to save the trained decoder model after the autoencoder training is complete.
- `--load_encoder` (str): Filename to load the encoder parameters from a pre-trained model.
- `--load_decoder` (str): Filename to load the decoder parameters from a pre-trained model.
- `--evaluate_encoder` (str): Filename for saving the encoder evaluation results on the input data.
- `--evaluate_decoder` (str): Filename for saving the decoder evaluation results on the input data.
- `--encoder_plugin` (str, default='default'): Name of the encoder plugin to use (e.g., CNN, ANN, LSTM, etc.).
- `--decoder_plugin` (str, default='default'): Name of the decoder plugin to use (e.g., CNN, ANN, LSTM, etc.).
- `--window_size` (int): Sliding window size for processing time-series data (applicable for time-series data processing).
- `--threshold_error` (float): Mean squared error (MSE) threshold to stop the training process once the error is below this value.
- `--initial_size` (int): Initial size of the encoder/decoder interface (i.e., latent space size) when starting the dimensionality reduction process.
- `--step_size` (int): Step size for adjusting the encoder/decoder interface during incremental or decremental search.
- `--remote_log` (str): URL of a remote API endpoint for saving debug variables and logging training results in JSON format.
- `--remote_load_config` (str): URL of a remote JSON configuration file that will be downloaded and executed during the process.
- `--remote_save_config` (str): URL of a remote API endpoint for saving the configuration in JSON format after it has been executed.
- `--username` (str): Username required for API endpoint authentication (if needed).
- `--password` (str): Password required for API endpoint authentication (if needed).
- `--load_config` (str): Path to load a configuration file from disk, containing all necessary parameters for the fusion and autoencoder process.
- `--save_config` (str): Path to save the current configuration after the process completes.
- `--save_log` (str): Path to save the current debug information (e.g., logs, intermediate results).
- `--quiet_mode` (flag): Suppresses output messages and minimizes terminal logging during execution.
- `--force_date` (flag): Force inclusion of the date in the output CSV files, typically for time-series data outputs.
- `--incremental_search` (flag): Enables or disables incremental search for adjusting the size of the encoder/decoder interface (latent space). If disabled, the system performs decremental search.
- `--headers` (flag): Indicates whether the CSV input files contain headers. If present, headers will be preserved in the output file.


### Examples of Use

#### Autoencoder Training Example

To train an autoencoder using the CNN encoder and decoder plugins with a window size of 128, use the following command:

```bash
feature-fusion.bat tests\data\merged_feature_data.csv --encoder_plugin cnn --decoder_plugin cnn --window_size 128
```

#### Evaluating Data with a pre-trained Encoder
To evaluate data using a pre-trained encoder model, use the following command:

```bash
feature-fusion.bat tests\data\merged_feature_data.csv --load_encoder encoder_model.h5
```

## Project Directory Structure
```md
feature-fusion/
│
├── app/                           # Main application package
│   ├── autoencoder_manager.py    # Manages autoencoder creation, training, saving, and loading
│   ├── cli.py                    # Handles command-line argument parsing
│   ├── config.py                 # Stores default configuration values
│   ├── config_handler.py         # Manages configuration loading, saving, and merging
│   ├── config_merger.py          # Merges configuration from various sources
│   ├── data_handler.py           # Handles data loading and saving
│   ├── data_processor.py         # Processes input data and runs the autoencoder pipeline
│   ├── main.py                   # Main entry point for the application
│   ├── plugin_loader.py          # Dynamically loads encoder and decoder plugins
│   ├── reconstruction.py         # Functionality for reconstructing data from autoencoder output
│   └── plugins/                       # Plugin directory
│       ├── decoder_plugin_ann.py         # Decoder plugin using an artificial neural network
│       ├── decoder_plugin_cnn.py         # Decoder plugin using a convolutional neural network
│       ├── decoder_plugin_lstm.py        # Decoder plugin using long short-term memory networks
│       ├── decoder_plugin_transformer.py # Decoder plugin using transformer layers
│       ├── encoder_plugin_ann.py         # Encoder plugin using an artificial neural network
│       ├── encoder_plugin_cnn.py         # Encoder plugin using a convolutional neural network
│       ├── encoder_plugin_lstm.py        # Encoder plugin using long short-term memory networks
│       └── encoder_plugin_transformer.py # Encoder plugin using transformer layers
│
├── tests/              # Test modules for the application
│   ├── acceptance          # User acceptance tests
│   ├── system              # System tests
│   ├── integration         # Integration tests
│   └── unit                # Unit tests
│
├── README.md                     # Overview and documentation for the project
├── requirements.txt              # Lists Python package dependencies
├── setup.py                      # Script for packaging and installing the project
├── set_env.bat                   # Batch script for environment setup
├── set_env.sh                    # Shell script for environment setup
└── .gitignore                         # Specifies intentionally untracked files to ignore
```

## Contributing

Contributions to the project are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to make contributions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
