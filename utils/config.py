import yaml
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_config(config_path=None):
    """
    Loads the YAML configuration file.
    
    Args:
        config_path (str): Path to the config file.
        
    Returns:
        dict: Configuration dictionary.
    """
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "config.yaml")
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config

def get_experiment_config(experiment_name, base_config_path=None):
    """
    Utility to potentially merge experiment specific overrides.
    """
    config = load_config(base_config_path)
    # Add experiment specific modifications here if needed
    config['experiment_name'] = experiment_name
    return config
