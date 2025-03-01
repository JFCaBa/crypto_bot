"""
Advanced Logging Configuration
===========================
Provides extended logging functionality beyond the basic setup in helpers.py.
Includes rotating file handlers, custom formatters, and log filtering.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

import loguru
from loguru import logger


class InterceptHandler(logging.Handler):
    """
    Intercept standard logging messages toward Loguru.
    
    This allows standard library logging to be captured by Loguru,
    ensuring all logs use the same format and destinations.
    """
    
    def emit(self, record):
        """
        Intercept logging messages.
        
        Args:
            record: Logging record
        """
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
            
        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
            
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


class LoggerManager:
    """
    Configure and manage logging with Loguru.
    
    Provides a centralized way to configure logging across the application,
    with support for multiple log sinks (file, console, etc.) and custom formats.
    """
    
    def __init__(
        self,
        log_level: str = "INFO",
        log_dir: str = "logs",
        app_name: str = "cryptobot",
        console_logging: bool = True,
        file_logging: bool = True,
        json_logging: bool = False,
        max_file_size: str = "100MB",
        rotation_time: str = "1 day",
        retention_time: str = "30 days",
        compression: str = "zip",
        error_log_separate: bool = True,
        intercept_std_lib: bool = True
    ):
        """
        Initialize the logger manager.
        
        Args:
            log_level: Default logging level
            log_dir: Directory for log files
            app_name: Application name for log file prefixes
            console_logging: Whether to log to console
            file_logging: Whether to log to files
            json_logging: Whether to format logs as JSON
            max_file_size: Maximum size of log files before rotation
            rotation_time: Time interval for log rotation
            retention_time: Time to keep old log files
            compression: Compression format for rotated logs
            error_log_separate: Whether to log errors to a separate file
            intercept_std_lib: Whether to intercept standard library logging
        """
        self.log_level = log_level
        self.log_dir = log_dir
        self.app_name = app_name
        self.console_logging = console_logging
        self.file_logging = file_logging
        self.json_logging = json_logging
        self.max_file_size = max_file_size
        self.rotation_time = rotation_time
        self.retention_time = retention_time
        self.compression = compression
        self.error_log_separate = error_log_separate
        self.intercept_std_lib = intercept_std_lib
        
        # Log IDs for each sink (needed for dynamic level changes)
        self.sink_ids = {}
        
        # Ensure log directory exists
        if self.file_logging and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # Message formats
        self.console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        
        self.file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        )
        
        self.json_format = lambda record: json.dumps({
            "timestamp": record["time"].strftime("%Y-%m-%d %H:%M:%S.%f"),
            "level": record["level"].name,
            "message": record["message"],
            "module": record["name"],
            "function": record["function"],
            "line": record["line"],
            "thread_name": record["extra"].get("thread_name", ""),
            "process_id": os.getpid()
        })
        
        # Initialize logging
        self.configure_logging()
        
    def configure_logging(self):
        """Configure Loguru logger."""
        # First, remove default logger
        logger.remove()
        
        # Add console logger if enabled
        if self.console_logging:
            console_id = logger.add(
                sys.stderr,
                level=self.log_level,
                format=self.console_format,
                colorize=True,
                backtrace=True,
                diagnose=True
            )
            self.sink_ids["console"] = console_id
        
        # Add file loggers if enabled
        if self.file_logging:
            # Main log file
            main_log_file = os.path.join(self.log_dir, f"{self.app_name}.log")
            main_id = logger.add(
                main_log_file,
                level=self.log_level,
                format=self.file_format if not self.json_logging else self.json_format,
                rotation=self.rotation_time,
                retention=self.retention_time,
                compression=self.compression,
                backtrace=True,
                diagnose=True
            )
            self.sink_ids["main"] = main_id
            
            # Error log file if separate
            if self.error_log_separate:
                error_log_file = os.path.join(self.log_dir, f"{self.app_name}_error.log")
                error_id = logger.add(
                    error_log_file,
                    level="ERROR",
                    format=self.file_format if not self.json_logging else self.json_format,
                    rotation=self.rotation_time,
                    retention=self.retention_time,
                    compression=self.compression,
                    backtrace=True,
                    diagnose=True,
                    filter=lambda record: record["level"].name == "ERROR" or record["level"].name == "CRITICAL"
                )
                self.sink_ids["error"] = error_id
        
        # Intercept standard library logging if enabled
        if self.intercept_std_lib:
            self._intercept_standard_logging()
            
        logger.info(f"Logger initialized with level {self.log_level}")
        
    def _intercept_standard_logging(self):
        """Configure standard library logging to use Loguru."""
        # Get all existing loggers
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
        
        # Replace all handlers with interceptor
        for name in logging.root.manager.loggerDict.keys():
            if name.startswith('uvicorn') or name.startswith('gunicorn'):
                # Skip uvicorn and gunicorn loggers to avoid conflicts
                continue
                
            logging.getLogger(name).handlers = [InterceptHandler()]
            logging.getLogger(name).propagate = False
            
    def set_level(self, level: str, sink_name: Optional[str] = None):
        """
        Change the logging level for a specific sink or all sinks.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            sink_name: Sink name to change level for (None for all)
        """
        if sink_name:
            if sink_name in self.sink_ids:
                logger.configure(levels=[{"name": level, "sink": self.sink_ids[sink_name]}])
                logger.info(f"Changed log level for sink '{sink_name}' to {level}")
            else:
                logger.warning(f"Sink '{sink_name}' not found")
        else:
            # Change level for all sinks
            for sink_id in self.sink_ids.values():
                logger.configure(levels=[{"name": level, "sink": sink_id}])
            
            logger.info(f"Changed log level for all sinks to {level}")
            
    def add_context(self, **kwargs):
        """
        Add context data to be included in all logs.
        
        Args:
            **kwargs: Key-value pairs to add to log context
        """
        logger.configure(extra=kwargs)
        logger.debug(f"Added context data: {kwargs}")
        
    def get_transaction_logger(self, transaction_id: str):
        """
        Create a logger that includes transaction ID in all logs.
        
        Args:
            transaction_id: Transaction ID to include in logs
            
        Returns:
            logger: Logger with transaction context
        """
        return logger.bind(transaction_id=transaction_id)
        
    def create_module_logger(self, module_name: str):
        """
        Create a logger for a specific module.
        
        Args:
            module_name: Module name
            
        Returns:
            logger: Module-specific logger
        """
        return logger.bind(name=module_name)


class PerformanceLogger:
    """
    Special logger for performance monitoring.
    
    Records execution time and resource usage for functions and code blocks.
    """
    
    def __init__(self, log_dir: str = "logs", app_name: str = "cryptobot"):
        """
        Initialize the performance logger.
        
        Args:
            log_dir: Directory for log files
            app_name: Application name for log file prefixes
        """
        self.log_dir = log_dir
        
        # Ensure log directory exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # Create performance log file
        perf_log_file = os.path.join(self.log_dir, f"{app_name}_performance.log")
        self.logger = logger.bind(performance=True)
        
        # Add performance sink
        self.perf_id = logger.add(
            perf_log_file,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[performance]}: {message}",
            filter=lambda record: record["extra"].get("performance", False),
            rotation="1 day",
            retention="30 days",
            compression="zip"
        )
        
        # Store timing data
        self.times = {}
        self.resource_usage = {}
        
    def start_timer(self, name: str):
        """
        Start a timer for measuring execution time.
        
        Args:
            name: Timer name
        """
        self.times[name] = time.time()
        self.logger.debug(f"Started timer: {name}")
        
    def stop_timer(self, name: str):
        """
        Stop a timer and log the execution time.
        
        Args:
            name: Timer name
            
        Returns:
            float: Execution time in seconds
        """
        if name not in self.times:
            self.logger.warning(f"Timer not found: {name}")
            return 0.0
            
        elapsed = time.time() - self.times[name]
        self.logger.info(f"{name} completed in {elapsed:.6f} seconds")
        
        # Store resource usage if available
        try:
            import resource
            self.resource_usage[name] = {
                'memory_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'cpu_time': resource.getrusage(resource.RUSAGE_SELF).ru_utime
            }
        except ImportError:
            pass
            
        return elapsed
        
    def log_execution_time(self, name: Optional[str] = None):
        """
        Decorator to log execution time of a function.
        
        Args:
            name: Custom name for the timer (defaults to function name)
            
        Returns:
            callable: Decorated function
        """
        def decorator(func):
            func_name = name or func.__name__
            
            def wrapper(*args, **kwargs):
                self.start_timer(func_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.stop_timer(func_name)
                    
            return wrapper
        return decorator
        
    def get_performance_report(self):
        """
        Generate a performance report.
        
        Returns:
            dict: Performance report
        """
        report = {
            'timers': {name: elapsed for name, elapsed in self.times.items()},
            'resource_usage': self.resource_usage
        }
        
        self.logger.info(f"Performance report: {json.dumps(report)}")
        return report


# Singleton instances
logger_manager = None
performance_logger = None

def setup_advanced_logger(
    log_level: str = "INFO",
    log_dir: str = "logs",
    app_name: str = "cryptobot",
    console_logging: bool = True,
    file_logging: bool = True,
    json_logging: bool = False
) -> LoggerManager:
    """
    Set up advanced logger with custom configuration.
    
    Args:
        log_level: Default logging level
        log_dir: Directory for log files
        app_name: Application name for log file prefixes
        console_logging: Whether to log to console
        file_logging: Whether to log to files
        json_logging: Whether to format logs as JSON
        
    Returns:
        LoggerManager: Logger manager instance
    """
    global logger_manager
    
    if logger_manager is None:
        logger_manager = LoggerManager(
            log_level=log_level,
            log_dir=log_dir,
            app_name=app_name,
            console_logging=console_logging,
            file_logging=file_logging,
            json_logging=json_logging
        )
        
    return logger_manager
    
def get_performance_logger(log_dir: str = "logs", app_name: str = "cryptobot") -> PerformanceLogger:
    """
    Get or create the performance logger.
    
    Args:
        log_dir: Directory for log files
        app_name: Application name for log file prefixes
        
    Returns:
        PerformanceLogger: Performance logger instance
    """
    global performance_logger
    
    if performance_logger is None:
        performance_logger = PerformanceLogger(log_dir, app_name)
        
    return performance_logger
    
def get_logger(module_name: str = None):
    """
    Get a logger instance.
    
    Args:
        module_name: Module name
        
    Returns:
        logger: Logger instance
    """
    if module_name:
        return logger.bind(module=module_name)
    return logger