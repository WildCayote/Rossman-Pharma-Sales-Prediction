import logging


def config_logger(log_file: str='../test.log'):
    logging.basicConfig(
        level=logging.INFO,  # Minimum log level to capture
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format with timestamp
        handlers=[
            logging.FileHandler(log_file),  # Output logs to a file
            logging.StreamHandler()  # Also output logs to the console
        ]
    )

def log_message(msg: str, level: str='INFO'):
    
    if level == "INFO": logging.info(msg)
    elif level == "DEBUG": logging.debug(msg)
    elif level == "ERROR": logging.error(msg)
    elif level == "WARNING": logging.warning(msg)
    else: logging.info(msg)