import pandas as pd
from app.reconstruction import unwindow_data

def load_csv_files(file_paths, validation_file_paths=None):
    """
    Load multiple CSV files and their corresponding validation files, ensuring that all 
    datasets have the same number of rows and columns, and that all data is numeric.
    
    Args:
        file_paths (list of str): List of file paths to the CSV files to be loaded.
        validation_file_paths (list of str, optional): List of file paths to validation CSV files.
            Defaults to None.
    
    Returns:
        tuple: 
            - datasets (list of pandas.DataFrame): List of loaded datasets.
            - validation_datasets (list of pandas.DataFrame): List of loaded validation datasets.
            
    Raises:
        ValueError: If any dataset contains non-numeric data, or if the datasets/validation datasets 
                    have mismatched row or column counts.
        Exception: If there is an error during file loading.
    """
    
    datasets = []  # Store main datasets
    validation_datasets = []  # Store validation datasets

    try:
        # Load each file from the main dataset file_paths
        for file_path in file_paths:
            # Load CSV assuming no headers and all columns are numeric
            data = pd.read_csv(file_path, header=None, sep=',')
            # Assign generic column names (col_0, col_1, ...)
            data.columns = [f'col_{i}' for i in range(len(data.columns))]

            # Check if all columns are numeric
            if not data.applymap(lambda x: isinstance(x, (int, float))).all().all():
                raise ValueError(f"Error: All columns in {file_path} must contain only numeric data.")
            
            datasets.append(data)  # Append to the datasets list

        # Ensure all datasets have the same number of rows and columns
        num_rows = datasets[0].shape[0]  # Number of rows in the first dataset
        num_cols = datasets[0].shape[1]  # Number of columns in the first dataset
        for i, dataset in enumerate(datasets):
            if dataset.shape[0] != num_rows or dataset.shape[1] != num_cols:
                raise ValueError(f"Error: Dataset {file_paths[i]} does not match the shape of the first dataset (rows: {num_rows}, columns: {num_cols}).")

        # Load validation datasets if validation file paths are provided
        if validation_file_paths:
            for validation_file_path in validation_file_paths:
                # Load CSV assuming no headers and all columns are numeric
                validation_data = pd.read_csv(validation_file_path, header=None, sep=',')
                # Assign generic column names
                validation_data.columns = [f'col_{i}' for i in range(len(validation_data.columns))]

                # Check if all columns in the validation dataset are numeric
                if not validation_data.applymap(lambda x: isinstance(x, (int, float))).all().all():
                    raise ValueError(f"Error: All columns in {validation_file_path} must contain only numeric data.")

                validation_datasets.append(validation_data)  # Append to validation datasets

            # Ensure all validation datasets have the same number of rows and columns
            validation_num_rows = validation_datasets[0].shape[0]  # Number of rows in the first validation dataset
            validation_num_cols = validation_datasets[0].shape[1]  # Number of columns in the first validation dataset
            for i, validation_dataset in enumerate(validation_datasets):
                if validation_dataset.shape[0] != validation_num_rows or validation_dataset.shape[1] != validation_num_cols:
                    raise ValueError(f"Error: Validation dataset {validation_file_paths[i]} does not match the shape of the first validation dataset (rows: {validation_num_rows}, columns: {validation_num_cols}).")

    except Exception as e:
        print(f"An error occurred while loading the CSV files: {e}")
        raise

    return datasets, validation_datasets


def write_csv(file_path, data, include_date=True, headers=True, window_size=None):
    """
    Write a pandas DataFrame to a CSV file, with options to include the date and headers.
    
    Args:
        file_path (str): The path to the CSV file to be written.
        data (pandas.DataFrame): The DataFrame to be written to the file.
        include_date (bool, optional): Whether to include the date in the output. Defaults to True.
        headers (bool, optional): Whether to include the column headers. Defaults to True.
        window_size (int, optional): Window size for the sliding window operation. Not used in this function. Defaults to None.
    
    Raises:
        Exception: If there is an error during file writing.
    """
    
    try:
        # If the date is to be included and the 'date' column exists, write CSV with index
        if include_date and 'date' in data.columns:
            data.to_csv(file_path, index=True, header=headers)
        else:
            # Otherwise, write CSV without the index
            data.to_csv(file_path, index=False, header=headers)
    except Exception as e:
        print(f"An error occurred while writing the CSV: {e}")
        raise
