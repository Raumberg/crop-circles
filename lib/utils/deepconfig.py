import os
import toml
import tabulate

from typing import Any, Dict, cast

__all__ = ["DeepConfig"]

class DeepConfig:
    def __init__(self, path: str = None) -> None:
        self.PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
        if path: 
            self.config_path = path
            self.config = self.load_config(path)
        else: 
            self.config: Dict[str, Any] = {}

    @staticmethod
    def load_from(config_filepath: str, append: bool = True) -> dict:
        """
        Load the configuration from a TOML file in a custom filepath
        """
        data = toml.load(config_filepath)
        if append:
            DeepConfig.config = cast(Dict[str, Any], data)
        return data
    
    def get(self, section: str, key: str) -> Any:
        """
        Get a value from the configuration.
        """
        return self.config.get(section, {}).get(key)
        
    def load_config(self) -> None:
        """
        Load the configuration from the TOML file.
        """
        try:
            self.config = toml.load(self.config_file)
            print("Configuration loaded successfully.")
        except FileNotFoundError:
            print(f"Error: The file '{self.config_file}' was not found.")
        except toml.TomlDecodeError as e:
            print(f"Error decoding TOML: {e}")
        
    def seek(self) -> None:
        ...

    @property
    def inspect(self) -> None:
        """
        Print the configuration as a table.
        """
        table_data = []
        for section, values in self.config.items():
            for key, value in cast(Dict[str, Any], values).items():
                table_data.append([section, key, value])

        print(tabulate(table_data, headers=["Section", "Key", "Value"], tablefmt="grid"))