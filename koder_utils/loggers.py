import json
import logging
import logging.config
from typing import List, Dict, Any


def setup_loggers(loggers: List[logging.Logger], default_level: int = logging.INFO, log_fname: str = None) -> None:
    sh = logging.StreamHandler()
    sh.setLevel(default_level)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    colored_formatter = logging.Formatter(log_format, datefmt="%H:%M:%S")
    sh.setFormatter(colored_formatter)
    handlers = [sh]

    if log_fname is not None:
        fh = logging.FileHandler(log_fname)
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format, datefmt="%H:%M:%S")
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        handlers.append(fh)

    for clogger in loggers:
        clogger.setLevel(logging.DEBUG)
        clogger.handlers = []
        clogger.addHandler(sh)

        for handler in handlers:
            clogger.addHandler(handler)

    root_logger = logging.getLogger()
    root_logger.handlers = []


def setup_logging(log_config_fname: str = None, log_file: str = None, log_level: str = None,
                  log_config_obj: Dict[str, Any] = None) -> None:
    if log_config_obj:
        assert not log_config_fname
        log_config = log_config_obj
    else:
        log_config = json.load(open(log_config_fname))

    if log_file is not None:
        log_config["handlers"]["log_file"]["filename"] = log_file

    if log_level is not None:
        log_config["handlers"]["console"]["level"] = log_level

    logging.config.dictConfig(log_config)
