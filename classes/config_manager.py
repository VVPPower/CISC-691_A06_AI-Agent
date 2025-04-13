import json
import logging
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ConfigManager:
    """Manages configuration settings from a JSON file."""
    config_file: Path

    def __post_init__(self):
        self.config_file = Path(self.config_file)
        if not self.config_file.exists():
            logging.error(f"Configuration file '{self.config_file}' not found.")
            exit(1)
        with open(self.config_file, "r") as file:
            self.settings = json.load(file)

    def get(self, key, default=None):
        """Safely fetch a configuration value."""
        return self.settings.get(key, default)

    def get_directory_names(self):
        dir_list = []
        for key, value in self.settings.items():
            if key.endswith("_directory"):
                dir_list.append(value)
        return dir_list

    def __str__(self):
        return str(self.settings)

    def to_dict(self):
        """Returns a dictionary representation of the settings."""
        return self.settings
