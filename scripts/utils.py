import logging


def config_logger(log_file: str='../test.log'):
    """
    Configures the logging system to write log messages to both a file and the console.

    The log messages include a timestamp, log level (INFO, DEBUG, ERROR, etc.), and the 
    message content. The log file is specified as a parameter, and both the log file and 
    the console will receive the same log output.
    
    Args:
        log_file: str, optional (default: "../test.log")
            The path to the file where logs should be saved. The default log file name is 'app.log'.
    """
    logging.basicConfig(
        level=logging.INFO,  # Minimum log level to capture
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format with timestamp
        handlers=[
            logging.FileHandler(log_file),  # Output logs to a file
            logging.StreamHandler()  # Also output logs to the console
        ]
    )

def log_message(msg: str, level: str='INFO'):
    """
    Logs a message at a specified log level (e.g., INFO, DEBUG, ERROR) with a timestamp.

    This function sends the log message to both the console and the configured log file.
    It provides an easy way to track events, errors, and important information in a consistent 
    format during the execution of a program.

    Args:
        message: str
            The log message content to be recorded.
    
        level: str, optional (default: "INFO")
            The severity level of the log message. Supported levels include:
            - "INFO" for general information (default)
            - "DEBUG" for detailed diagnostic information
            - "ERROR" for error messages and exceptions
            - "WARNING" for warnings about potential issues
    """
    if level == "INFO": logging.info(msg)
    elif level == "DEBUG": logging.debug(msg)
    elif level == "ERROR": logging.error(msg)
    elif level == "WARNING": logging.warning(msg)
    else: logging.info(msg)