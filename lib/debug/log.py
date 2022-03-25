"""Debug logging
"""
import os
import json
import logging
import logging.config

debug_path = os.getcwd()

with open(f"{debug_path}/lib/debug/logging.json", "rt") as file:
    config = json.load(file)
# LOG_FORMAT = "[%(asctime)-10s] (line %(lineno)d) %(name)s:%(levelname)s - %(message)s"
LOG_FORMAT = "[%(filename)s, line %(lineno)d) %(name)s:%(levelname)s - %(message)s"

logging.config.dictConfig(config)


def maker_logger(name=None):
    # 1 logger instance
    logger = logging.getLogger(name)

    # 2 logger level > The lowest level DEBUG
    logger.setLevel(logging.DEBUG)

    # 3 Set formatter
    formatter = logging.Formatter(LOG_FORMAT)

    # 4 Create the handelr instance
    console = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=f"{debug_path}test.log")

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


PRINTER = maker_logger()

if __name__ == "__main__":
    # _logger = maker_logger()

    PRINTER.debug("test")
    PRINTER.info("test")
    PRINTER.warning("test")
