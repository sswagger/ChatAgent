"""
Configuration loader for the AI Chat Agent.
Handles loading and saving configuration files.
"""

import os
import json
import threading


def load_config(config_file: str, default_config: dict) -> dict:
    """Load user config from file, fall back to defaults."""
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                configurations = json.load(f)
                for key, value in default_config.items():
                    if key not in configurations:
                        configurations[key] = value
        except (json.JSONDecodeError, IOError):
            configurations = default_config.copy()
            save_config(configurations, config_file)
    else:
        configurations = default_config.copy()
        save_config(configurations, config_file)
    
    return configurations


def save_config(configuration: dict, config_file: str):
    """Save config to file."""
    with threading.Lock():
        with open(config_file, 'w') as f:
            json.dump(configuration, f, indent=4)
