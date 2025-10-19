"""
Logging setup for Trajectory Synthetic Data Generator.
Uses loguru for beautiful, structured logging.
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


class LoggerSetup:
    """Setup and configure application logging."""
    
    def __init__(self, log_level: str = "INFO", log_file: Optional[str] = None):
        """
        Initialize logger setup.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file. If None, logs only to console.
        """
        self.log_level = log_level.upper()
        self.log_file = log_file
        self._configure_logger()
    
    def _configure_logger(self):
        """Configure loguru logger with custom format and handlers."""
        # Remove default handler
        logger.remove()
        
        # Console handler with colors
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=self.log_level,
            colorize=True
        )
        
        # File handler (if specified)
        if self.log_file:
            # Ensure log directory exists
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                self.log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                level=self.log_level,
                rotation="10 MB",  # Rotate when file reaches 10MB
                retention="7 days",  # Keep logs for 7 days
                compression="zip",  # Compress rotated logs
                enqueue=True  # Thread-safe logging
            )
    
    @staticmethod
    def get_logger(name: str = __name__):
        """
        Get a logger instance.
        
        Args:
            name: Logger name (typically __name__ of the calling module)
            
        Returns:
            Logger instance
        """
        return logger.bind(name=name)


def setup_logger(log_level: str = "INFO", log_file: Optional[str] = None) -> logger:
    """
    Setup and return configured logger.
    
    Args:
        log_level: Logging level
        log_file: Path to log file
        
    Returns:
        Configured logger instance
    """
    LoggerSetup(log_level, log_file)
    return logger


# Convenience function for getting logger in modules
def get_logger(name: str = __name__):
    """
    Get logger for a module.
    
    Args:
        name: Module name (use __name__)
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


if __name__ == "__main__":
    # Test logging
    test_logger = setup_logger("DEBUG", "logs/test.log")
    
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.success("This is a success message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    
    print("\n✅ Logger test completed. Check logs/test.log")