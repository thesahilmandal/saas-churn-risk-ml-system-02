import os
import sys
import yaml, json
import pickle
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from typing import Any, Dict

from src.exception import CustomerChurnException
from src.logging import logging


# ==============================
# Internal Helpers
# ==============================

def _prepare_file_path(file_path: str, replace: bool = True) -> None:
    """
    Prepare file path by removing existing file (if replace=True)
    and creating parent directories.
    """
    try:
        if replace and os.path.exists(file_path):
            logging.info(f"Removing existing file: {file_path}")
            os.remove(file_path)

        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    except Exception as e:
        raise CustomerChurnException(e, sys)


# ==============================
# YAML Utilities
# ==============================

def read_yaml_file(file_path: str) -> Dict[str, Any]:
    """
    Read YAML file and return contents as dictionary.
    """
    try:
        logging.info(f"Reading YAML file: {file_path}")
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise CustomerChurnException(e, sys)


def write_yaml_file(file_path: str, content: Dict[str, Any], replace: bool = True) -> None:
    """
    Write dictionary content to YAML file.
    """
    try:
        logging.info(f"Writing YAML file: {file_path}")
        _prepare_file_path(file_path, replace)

        with open(file_path, "w") as file:
            yaml.dump(content, file, sort_keys=False)

    except Exception as e:
        raise CustomerChurnException(e, sys)


# ==============================
# NumPy Utilities
# ==============================

def save_numpy_array_data(
    file_path: str,
    array: np.ndarray,
    replace: bool = True
) -> None:
    """
    Save NumPy array to disk.
    """
    try:
        logging.info(f"Saving NumPy array: {file_path}")
        _prepare_file_path(file_path, replace)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise CustomerChurnException(e, sys)


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Load NumPy array from disk.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Numpy file not found: {file_path}")

        logging.info(f"Loading NumPy array: {file_path}")
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)

    except Exception as e:
        raise CustomerChurnException(e, sys)


# ==============================
# Pickle Utilities
# ==============================

def save_object(file_path: str, obj: Any, replace: bool = True) -> None:
    """
    Serialize and save Python object using pickle.
    """
    try:
        logging.info(f"Saving object: {file_path}")
        _prepare_file_path(file_path, replace)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomerChurnException(e, sys)


def load_object(file_path: str) -> Any:
    """
    Load serialized Python object from disk.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Object file not found: {file_path}")

        logging.info(f"Loading object: {file_path}")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomerChurnException(e, sys)


# ==============================
# JSON Utilities
# ==============================

def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    Read JSON file and return contents as a dictionary.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: Parsed JSON content.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        logging.info(f"Reading JSON file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception as e:
        raise CustomerChurnException(e, sys)


def write_json_file(
    file_path: str,
    content: Dict[str, Any],
    replace: bool = True
) -> None:
    """
    Write dictionary content to a JSON file.

    Args:
        file_path (str): Path to the JSON file.
        content (Dict[str, Any]): Data to be written.
        replace (bool): Whether to overwrite existing file.
    """
    try:
        logging.info(f"Writing JSON file: {file_path}")
        _prepare_file_path(file_path, replace)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(content, file, indent=4)

    except Exception as e:
        raise CustomerChurnException(e, sys)


# ==============================
# CSV Utilities
# ==============================

def read_csv_file(
    file_path: str,
    **kwargs
) -> pd.DataFrame:
    """
    Read a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.
        **kwargs: Additional arguments forwarded to pandas.read_csv().

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        logging.info(f"Reading CSV file: {file_path}")
        df = pd.read_csv(file_path, **kwargs)

        logging.info(f"CSV loaded successfully | Shape: {df.shape}")
        return df

    except Exception as e:
        raise CustomerChurnException(e, sys)


def save_csv_file(
    file_path: str,
    df: pd.DataFrame,
    replace: bool = True,
    index: bool = False,
    **kwargs
) -> None:
    """
    Save a pandas DataFrame to CSV.

    Args:
        file_path (str): Output CSV path.
        df (pd.DataFrame): DataFrame to save.
        replace (bool): Whether to overwrite existing file.
        index (bool): Whether to write row index.
        **kwargs: Additional arguments forwarded to pandas.to_csv().
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input object must be a pandas DataFrame")

        logging.info(f"Saving CSV file: {file_path}")
        _prepare_file_path(file_path, replace)

        df.to_csv(file_path, index=index, **kwargs)

        logging.info(f"CSV saved successfully | Path: {file_path}")

    except Exception as e:
        raise CustomerChurnException(e, sys)


# ==============================
# Parquet Utilities
# ==============================

def csv_to_parquet(
    csv_path: str,
    parquet_path: str,
    replace: bool = True,
    **read_kwargs
) -> None:
    """
    Convert a CSV file to Parquet format.

    Args:
        csv_path (str): Input CSV file path.
        parquet_path (str): Output Parquet file path.
        replace (bool): Whether to overwrite existing Parquet file.
        **read_kwargs: Additional args for pandas.read_csv().
    """
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        logging.info(f"Converting CSV to Parquet | Source: {csv_path}")

        _prepare_file_path(parquet_path, replace)

        df = pd.read_csv(csv_path, **read_kwargs)

        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_path)

        logging.info(f"Parquet file created successfully | Path: {parquet_path}")

    except Exception as e:
        raise CustomerChurnException(e, sys)


def parquet_to_csv(
    parquet_path: str,
    csv_path: str,
    replace: bool = True,
    index: bool = False,
    **kwargs
) -> None:
    """
    Convert a Parquet file to CSV format.

    Args:
        parquet_path (str): Input Parquet file path.
        csv_path (str): Output CSV file path.
        replace (bool): Whether to overwrite existing CSV file.
        index (bool): Whether to write row index.
        **kwargs: Additional args for pandas.to_csv().
    """
    try:
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        logging.info(f"Converting Parquet to CSV | Source: {parquet_path}")

        _prepare_file_path(csv_path, replace)

        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        df.to_csv(csv_path, index=index, **kwargs)

        logging.info(f"CSV file created successfully | Path: {csv_path}")

    except Exception as e:
        raise CustomerChurnException(e, sys)