"""
Professional logging system for RC Element Prediction Project
Provides colored, emoji-enhanced logging with YAML configuration.
"""

import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class EmojiFormatter(logging.Formatter):
    """Custom formatter with emojis and colors for different log levels."""

    # Emojis for different modules
    MODULE_EMOJIS = {
        "DATA": "📁",
        "GRAPH": "🏗️",
        "MODEL": "🧠",
        "TRAIN": "🚀",
        "EVAL": "📊",
        "SYSTEM": "⚙️",
        "DEBUG": "🔍",
    }

    # Colors for log levels (ANSI codes)
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
        "RESET": "\033[0m",  # Reset
    }

    # Level emojis
    LEVEL_EMOJIS = {
        "DEBUG": "🔍",
        "INFO": "ℹ️ ",
        "WARNING": "⚠️ ",
        "ERROR": "❌",
        "CRITICAL": "💥",
    }

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: bool = True,
        use_emojis: bool = True,
    ):
        default_fmt = "%(asctime)s | %(emoji)s %(name)-6s | %(level_emoji)s %(levelname)-8s | %(message)s"
        super().__init__(fmt or default_fmt, datefmt)
        self.use_colors = use_colors
        self.use_emojis = use_emojis

    def format(self, record: logging.LogRecord) -> str:
        # Add emoji based on logger name
        if self.use_emojis:
            record.emoji = ""
            for module, emoji in self.MODULE_EMOJIS.items():
                if module in record.name:
                    record.emoji = emoji
                    break
            else:
                # Fallback: use first 3 letters of name
                record.emoji = f"[{record.name[:3]}]"

            # Add level emoji
            record.level_emoji = self.LEVEL_EMOJIS.get(record.levelname, "")
        else:
            record.emoji = ""
            record.level_emoji = ""

        # Format the message
        message = super().format(record)

        # Apply colors if enabled
        if self.use_colors and record.levelname in self.COLORS:
            message = f"{self.COLORS[record.levelname]}{message}{self.COLORS['RESET']}"

        return message


class ProjectLogger:
    """Main logger class that reads configuration from YAML."""

    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = "../configs/logger.yaml"):
        """Initialize logger with YAML configuration."""
        if not hasattr(self, "_initialized"):
            self.config = self._load_config(config_path)
            self.loggers: Dict[str, logging.Logger] = {}
            self._setup_loggers()
            self._initialized = True

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load logger configuration from YAML."""
        config_file = Path(config_path)

        if not config_file.exists():
            print(f"⚠️  Logger config not found at {config_path}, using defaults")
            return self._get_default_config()

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config.get("logger", {})
        except Exception as e:
            print(f"⚠️  Failed to load logger config: {e}, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "level": "INFO",
            "use_colors": True,
            "format": "%(asctime)s | %(emoji)s %(name)-6s | %(level_emoji)s %(levelname)-8s | %(message)s",
            "date_format": "%H:%M:%S",
            "modules": {
                "data_manager": {"level": "INFO", "name": "DATA"},
                "graph_builder": {"level": "INFO", "name": "GRAPH"},
                "model": {"level": "INFO", "name": "MODEL"},
                "training": {"level": "INFO", "name": "TRAIN"},
                "evaluation": {"level": "INFO", "name": "EVAL"},
                "system": {"level": "INFO", "name": "SYSTEM"},
            },
        }

    def _setup_loggers(self):
        """Setup all loggers based on configuration."""
        # Get global settings
        global_level = getattr(logging, self.config.get("level", "INFO"))
        use_colors = self.config.get("use_colors", True)

        # Create formatter
        formatter = EmojiFormatter(
            fmt=self.config.get(
                "format",
                "%(asctime)s | %(emoji)s %(name)-6s | %(level_emoji)s %(levelname)-8s | %(message)s",
            ),
            datefmt=self.config.get("date_format", "%H:%M:%S"),
            use_colors=use_colors,
            use_emojis=True,
        )

        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(global_level)

        # Setup file handler if enabled
        file_handler = None
        file_config = self.config.get("file_logging", {})
        if file_config.get("enabled", False):
            try:
                from logging.handlers import RotatingFileHandler

                log_dir = Path(file_config.get("path", "logs"))
                log_dir.mkdir(exist_ok=True)

                filename = file_config.get("filename", "project_{date}.log")
                filename = filename.replace("{date}", datetime.now().strftime("%Y%m%d"))

                file_handler = RotatingFileHandler(
                    log_dir / filename,
                    maxBytes=file_config.get("max_size_mb", 10) * 1024 * 1024,
                    backupCount=file_config.get("backup_count", 5),
                )
                file_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
                    )
                )
            except Exception as e:
                self._log_to_console(f"⚠️  Failed to setup file logging: {e}")

        # Create loggers for each module
        modules = self.config.get("modules", {})

        for module_key, module_config in modules.items():
            logger_name = module_config.get("name", module_key.upper())
            logger_level = getattr(logging, module_config.get("level", "INFO"))

            # Create logger
            logger = logging.getLogger(logger_name)
            logger.setLevel(logger_level)

            # Clear existing handlers
            logger.handlers.clear()

            # Add console handler
            logger.addHandler(console_handler)

            # Add file handler if available
            if file_handler:
                logger.addHandler(file_handler)

            # Prevent propagation
            logger.propagate = False

            # Store logger
            self.loggers[module_key] = logger

    def _log_to_console(self, message: str):
        """Log directly to console (used during initialization)."""
        print(message)

    def get_logger(self, module: str) -> logging.Logger:
        """
        Get logger for a specific module.

        Args:
            module: Module name (data_manager, graph_builder, model, etc.)

        Returns:
            Configured logger instance
        """
        if module in self.loggers:
            return self.loggers[module]
        else:
            # Return a default logger if module not configured
            default_logger = logging.getLogger(module.upper())
            default_logger.setLevel(logging.INFO)

            # Add handler if needed
            if not default_logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                handler.setFormatter(EmojiFormatter(use_colors=True, use_emojis=True))
                default_logger.addHandler(handler)
                default_logger.propagate = False

            return default_logger

    def update_config(self, config_updates: Dict[str, Any]):
        """
        Update logger configuration dynamically.

        Args:
            config_updates: Dictionary with configuration updates
        """

        def deep_update(source: Dict, updates: Dict):
            for key, value in updates.items():
                if (
                    isinstance(value, dict)
                    and key in source
                    and isinstance(source[key], dict)
                ):
                    deep_update(source[key], value)
                else:
                    source[key] = value

        deep_update(self.config, config_updates)
        self._setup_loggers()


# Global logger instance
_logger = ProjectLogger()


# Convenience functions for common modules
def get_data_logger() -> logging.Logger:
    """Get logger for data management operations."""
    return _logger.get_logger("data_manager")


def get_graph_logger() -> logging.Logger:
    """Get logger for graph building operations."""
    return _logger.get_logger("graph_builder")


def get_model_logger() -> logging.Logger:
    """Get logger for model operations."""
    return _logger.get_logger("model")


def get_train_logger() -> logging.Logger:
    """Get logger for training operations."""
    return _logger.get_logger("training")


def get_eval_logger() -> logging.Logger:
    """Get logger for evaluation operations."""
    return _logger.get_logger("evaluation")


def get_system_logger() -> logging.Logger:
    """Get logger for system operations."""
    return _logger.get_logger("system")


def get_logger(name: str) -> logging.Logger:
    """Get logger by name."""
    return _logger.get_logger(name)
