"""I/O utilities for saving and loading results"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd


def create_output_path(
    base_dir: str,
    attack_name: str,
    model_name: str,
    dataset_name: str,
    config: Optional[Dict[str, Any]] = None,
    extension: str = "jsonl"
) -> Path:
    """
    Create output file path with timestamp and parameters

    Args:
        base_dir: Base directory for outputs
        attack_name: Name of attack method
        model_name: Name of target model
        dataset_name: Name of dataset
        config: Configuration dictionary (optional)
        extension: File extension

    Returns:
        Path object for output file
    """
    # Create base directory
    output_dir = Path(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract model short name
    model_short = model_name.split("/")[-1].replace("-", "_")

    # Build filename
    filename_parts = [
        attack_name,
        model_short,
        dataset_name,
        datetime.now().strftime("%Y%m%d_%H%M%S")
    ]

    # Add config parameters if provided
    if config:
        if "temperature" in config:
            filename_parts.append(f"T{config['temperature']}")
        if "num_samples" in config:
            filename_parts.append(f"S{config['num_samples']}")
        if "num_iterations" in config:
            filename_parts.append(f"I{config['num_iterations']}")

    filename = "_".join(filename_parts) + f".{extension}"
    return output_dir / filename


def save_jsonl(data: List[Dict[str, Any]], filepath: str):
    """
    Save data to JSONL file

    Args:
        data: List of dictionaries to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file

    Args:
        filepath: Input file path

    Returns:
        List of dictionaries
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2):
    """
    Save data to JSON file

    Args:
        data: Dictionary to save
        filepath: Output file path
        indent: JSON indentation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file

    Args:
        filepath: Input file path

    Returns:
        Dictionary
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(data: List[Dict[str, Any]], filepath: str):
    """
    Save data to CSV file

    Args:
        data: List of dictionaries to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False, encoding="utf-8")


def load_csv(filepath: str) -> List[Dict[str, Any]]:
    """
    Load data from CSV file

    Args:
        filepath: Input file path

    Returns:
        List of dictionaries
    """
    df = pd.read_csv(filepath, encoding="utf-8")
    return df.to_dict("records")


class ResultLogger:
    """Logger for attack results with automatic saving"""

    def __init__(self, filepath: str, auto_save: bool = True):
        """
        Initialize result logger

        Args:
            filepath: Output file path
            auto_save: Whether to auto-save after each append
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save
        self.results = []

        # Load existing results if file exists
        if self.filepath.exists():
            try:
                self.results = load_jsonl(str(self.filepath))
            except:
                self.results = []

    def append(self, result: Dict[str, Any]):
        """
        Append a result

        Args:
            result: Result dictionary
        """
        self.results.append(result)

        if self.auto_save:
            self.save()

    def extend(self, results: List[Dict[str, Any]]):
        """
        Extend results with multiple items

        Args:
            results: List of result dictionaries
        """
        self.results.extend(results)

        if self.auto_save:
            self.save()

    def save(self):
        """Save all results to file"""
        save_jsonl(self.results, str(self.filepath))

    def __len__(self):
        return len(self.results)

    def __getitem__(self, idx):
        return self.results[idx]
