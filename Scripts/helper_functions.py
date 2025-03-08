import logging
import os

# Define log file path
LOG_PATH = "D:/MLOPS-2025-DSC/mlops2025-DSC/Logs"
os.makedirs(LOG_PATH, exist_ok=True)
LOG_FILE = os.path.join(LOG_PATH, "mlops_training.log")

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_logger():
    """
    Returns the configured logger.
    """
    return logging.getLogger()

def log_info(message):
    """
    Logs an info-level message.
    """
    logger = get_logger()
    logger.info(message)
    print(f"INFO: {message}")  # Optional console output

def log_error(message):
    """
    Logs an error-level message.
    """
    logger = get_logger()
    logger.error(message)
    print(f"ERROR: {message}")  # Optional console output

def log_warning(message):
    """
    Logs a warning-level message.
    """
    logger = get_logger()
    logger.warning(message)
    print(f"WARNING: {message}")  # Optional console output