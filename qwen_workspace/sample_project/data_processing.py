"""
Data processing module for the Sample Project.

Contains functions for reading, cleaning, and transforming data.
"""

import csv
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Global variable to store processed data cache
# FIXME: This global cache might cause issues in multi-threaded environments.
_data_cache: Dict[str, List[Dict[str, Any]]] = {}

def read_csv_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Read data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries representing rows.
    """
    data = []
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        logger.info(f"Successfully read {len(data)} rows from {file_path}")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
    return data

def clean_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean the data by removing empty rows and standardizing fields.
    
    Args:
        data (List[Dict[str, Any]]): Raw data list.
        
    Returns:
        List[Dict[str, Any]]: Cleaned data list.
    """
    cleaned_data = []
    for row in data:
        # Remove rows where all values are empty
        if any(value.strip() for value in row.values()):
            # Standardize string fields
            cleaned_row = {k: v.strip() if isinstance(v, str) else v for k, v in row.items()}
            cleaned_data.append(cleaned_row)
            
    logger.info(f"Cleaned data: {len(data)} -> {len(cleaned_data)} rows")
    return cleaned_data

def transform_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform data by adding calculated fields.
    
    Args:
        data (List[Dict[str, Any]]): Cleaned data list.
        
    Returns:
        List[Dict[str, Any]]: Transformed data list.
    """
    transformed_data = []
    for row in data:
        new_row = row.copy()
        # Example transformation: calculate a combined field
        if 'first_name' in row and 'last_name' in row:
            new_row['full_name'] = f"{row['first_name']} {row['last_name']}"
        
        # Example transformation: convert a field to integer
        if 'age' in row:
            try:
                new_row['age'] = int(row['age'])
            except ValueError:
                new_row['age'] = None
                logger.warning(f"Could not convert age '{row['age']}' to integer.")
        
        transformed_data.append(new_row)
        
    logger.info(f"Transformed {len(data)} rows of data.")
    return transformed_data

def process_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Main data processing function that orchestrates reading, cleaning, and transforming.
    
    This is a high-level function that combines all processing steps.
    
    Args:
        file_path (str): Path to the input data file (CSV).
        
    Returns:
        List[Dict[str, Any]]: Fully processed data list.
    """
    # Check cache first
    if file_path in _data_cache:
        logger.info("Returning data from cache.")
        return _data_cache[file_path]
    
    # Read data
    raw_data = read_csv_data(file_path)
    if not raw_data:
        return []
    
    # Clean data
    cleaned_data = clean_data(raw_data)
    
    # Transform data
    final_data = transform_data(cleaned_data)
    
    # Cache the result
    _data_cache[file_path] = final_data
    
    return final_data

# Duplicate function for testing duplicate detection
def process_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Duplicate of the main data processing function.
    This is intentionally duplicated to test the duplicate code detection feature.
    
    Args:
        file_path (str): Path to the input data file (CSV).
        
    Returns:
        List[Dict[str, Any]]: Fully processed data list.
    """
    # Check cache first
    if file_path in _data_cache:
        logger.info("Returning data from cache.")
        return _data_cache[file_path]
    
    # Read data
    raw_data = read_csv_data(file_path)
    if not raw_data:
        return []
    
    # Clean data
    cleaned_data = clean_data(raw_data)
    
    # Transform data
    final_data = transform_data(cleaned_data)
    
    # Cache the result
    _data_cache[file_path] = final_data
    
    return final_data