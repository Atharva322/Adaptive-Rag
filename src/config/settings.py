"""
Configuration settings for the application.
"""

import os
from pathlib import Path

import yaml

SEARCH_TYPE: str = "hybrid"       # "similarity" | "mmr" | "hybrid"
RETRIEVER_K: int = 8
BM25_WEIGHT: float = 0.5
VECTOR_WEIGHT: float = 0.5
MMR_FETCH_K: int = 20
MMR_LAMBDA_MULT: float = 0.5


class Config:
    """Load and manage configuration from YAML file."""

    def __init__(self, config_file: str = None):
        """
        Initialize configuration from YAML file.

        Args:
            config_file: Optional path to config file. Defaults to prompts.yaml.
        """
        base_path = Path(__file__).parent
        config_path = (
            base_path / "prompts.yaml"
            if config_file is None
            else Path(config_file)
        )
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def prompt(self, key: str) -> str:
        """
        Retrieve a prompt from configuration.

        Args:
            key: The prompt key.

        Returns:
            The prompt template string.
        """
        return self.config["prompts"][key]
