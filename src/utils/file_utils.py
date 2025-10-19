"""
File utility functions for Trajectory Synthetic Data Generator.
"""

import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from .logger import get_logger

logger = get_logger(__name__)


class FileHandler:
    """Handle file operations for the project."""
    
    @staticmethod
    def ensure_dir(directory: Union[str, Path]) -> Path:
        """
        Ensure directory exists, create if it doesn't.
        
        Args:
            directory: Directory path
            
        Returns:
            Path object
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")
        return dir_path
    
    @staticmethod
    def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Read JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Read JSON file: {file_path}")
        return data
    
    @staticmethod
    def write_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2):
        """
        Write data to JSON file.
        
        Args:
            data: Data to write
            file_path: Path to output file
            indent: JSON indentation (default: 2)
        """
        file_path = Path(file_path)
        FileHandler.ensure_dir(file_path.parent)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.info(f"Wrote JSON file: {file_path}")
    
    @staticmethod
    def read_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Read JSONL (JSON Lines) file.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of parsed JSON objects
        """
        data = []
        with jsonlines.open(file_path, 'r') as reader:
            for obj in reader:
                data.append(obj)
        logger.debug(f"Read JSONL file: {file_path} ({len(data)} lines)")
        return data
    
    @staticmethod
    def write_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path]):
        """
        Write data to JSONL (JSON Lines) file.
        
        Args:
            data: List of dictionaries to write
            file_path: Path to output file
        """
        file_path = Path(file_path)
        FileHandler.ensure_dir(file_path.parent)
        
        with jsonlines.open(file_path, 'w') as writer:
            writer.write_all(data)
        logger.info(f"Wrote JSONL file: {file_path} ({len(data)} lines)")
    
    @staticmethod
    def append_jsonl(data: Union[Dict[str, Any], List[Dict[str, Any]]], 
                     file_path: Union[str, Path]):
        """
        Append data to JSONL file.
        
        Args:
            data: Dictionary or list of dictionaries to append
            file_path: Path to output file
        """
        file_path = Path(file_path)
        FileHandler.ensure_dir(file_path.parent)
        
        with jsonlines.open(file_path, 'a') as writer:
            if isinstance(data, dict):
                writer.write(data)
            else:
                writer.write_all(data)
        
        count = 1 if isinstance(data, dict) else len(data)
        logger.debug(f"Appended {count} line(s) to JSONL file: {file_path}")
    
    @staticmethod
    def read_text(file_path: Union[str, Path]) -> str:
        """
        Read text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            File contents as string
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.debug(f"Read text file: {file_path} ({len(content)} chars)")
        return content
    
    @staticmethod
    def write_text(content: str, file_path: Union[str, Path]):
        """
        Write text to file.
        
        Args:
            content: Text content to write
            file_path: Path to output file
        """
        file_path = Path(file_path)
        FileHandler.ensure_dir(file_path.parent)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Wrote text file: {file_path} ({len(content)} chars)")
    
    @staticmethod
    def list_files(directory: Union[str, Path], 
                   pattern: str = "*", 
                   recursive: bool = False) -> List[Path]:
        """
        List files in directory.
        
        Args:
            directory: Directory to search
            pattern: File pattern (default: "*")
            recursive: Search recursively (default: False)
            
        Returns:
            List of file paths
        """
        dir_path = Path(directory)
        
        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))
        
        # Filter to only files (not directories)
        files = [f for f in files if f.is_file()]
        
        logger.debug(f"Found {len(files)} files in {directory} (pattern: {pattern})")
        return files
    
    @staticmethod
    def get_project_root() -> Path:
        """
        Get project root directory.
        
        Returns:
            Path to project root
        """
        # Go up from src/utils/ to project root
        return Path(__file__).parent.parent.parent


# Convenience functions
def ensure_dir(directory: Union[str, Path]) -> Path:
    """Ensure directory exists."""
    return FileHandler.ensure_dir(directory)


def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Read JSON file."""
    return FileHandler.read_json(file_path)


def write_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2):
    """Write JSON file."""
    FileHandler.write_json(data, file_path, indent)


def read_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read JSONL file."""
    return FileHandler.read_jsonl(file_path)


def write_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path]):
    """Write JSONL file."""
    FileHandler.write_jsonl(data, file_path)


if __name__ == "__main__":
    # Test file operations
    from .logger import setup_logger
    setup_logger("DEBUG")
    
    test_dir = FileHandler.get_project_root() / "data" / "test"
    
    # Test JSON
    test_data = {"test": "data", "number": 123}
    write_json(test_data, test_dir / "test.json")
    read_data = read_json(test_dir / "test.json")
    print(f" JSON test: {read_data}")
    
    # Test JSONL
    test_list = [{"id": 1}, {"id": 2}, {"id": 3}]
    write_jsonl(test_list, test_dir / "test.jsonl")
    read_list = read_jsonl(test_dir / "test.jsonl")
    print(f" JSONL test: {len(read_list)} items")
    
    print("\n File utilities test completed")