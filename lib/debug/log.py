"""Debug logging
"""
import os
import json
import logging
import logging.config

curr_path = os.getcwd()
# NEED: Based on run file in terminal, enter the path of the debugging library folder
path_lib = "/lib/debug/"
path_lib = curr_path + path_lib
if path_lib == curr_path:
    raise Exception("Please enter the path of the debugging library folder")

with open(f"{path_lib}/logging.json", "rt") as file:
    config = json.load(file)

# LOG_FORMAT = "[%(asctime)-10s] (line %(lineno)d) %(name)s:%(levelname)s - %(message)s"
LOG_FORMAT = "[%(filename)s, line %(lineno)d) %(name)s:%(levelname)s - %(message)s"

logging.config.dictConfig(config)


def maker_logger(name="None", mode=None, file_path=None, file_mode="a"):
    """
    """
    # 1 logger instance
    logger = logging.getLogger(name)

    # 2 logger level > The lowest level DEBUG
    logger.setLevel(logging.DEBUG)

    # 3 Set formatter
    formatter = logging.Formatter(LOG_FORMAT)

    # 4 Create the handelr instance
    console = logging.StreamHandler()
    if file_path:
        file_handler = logging.FileHandler(filename=file_path)
    else:
        file_handler = logging.FileHandler(filename=f"{path_lib}/example.log")

    # 5 Set different level to each handler
    console.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    # 6 Set output format of handler
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 7 Add handler to logger
    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    _logger = maker_logger()
    _logger.debug("test")
    _logger.info("test")
    _logger.warning("test")
