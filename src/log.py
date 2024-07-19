import functools
import logging
import logging.handlers
import os
import socket
import traceback
from configparser import ConfigParser
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import logging_loki
from rich.logging import RichHandler

logging_loki.emitter.LokiEmitter.level_tag = (
    "level"  # assign to a variable named handler
)

HOST = f"{os.getlogin()}@{socket.gethostname()}"


def get_logger(config: ConfigParser, name: str, email_alert: bool = False):
    """Get logger with logs that are pretty printed to console, locally stored and
    optionally streamed to Grafana-Loki"""

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(get_stream_handler())
    # Send logs to a local file
    if config.getboolean("logging", "send_to_file"):
        logger.addHandler(get_file_handler(config))
    # Send logs to Grafana-Loki
    if config.getboolean("logging", "send_to_loki"):
        logger.addHandler(get_stream_loki_handler(config))
    if email_alert:
        logger.addHandler(get_email_handler(config))

    return logger


def get_file_handler(config: ConfigParser):
    format = str(
        {
            "time": "%(asctime)s",
            "level": "%(levelname)s",
            "app": config["logging"]["APP_NAME"],
            "uuid": config["logging"]["UIID"],
            "host": HOST,
            "logger": "%(name)s",
            "location": "%(filename)s:%(lineno)d",
            "message": "%(message)s",
        }
    )
    logfile = str(Path(config["paths"]["ROOT"]).joinpath(Path(config["paths"]["LOGS"])))
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(format))
    return file_handler


def get_stream_handler():
    format = f"%(name)s - %(message)s"
    stream_handler = RichHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(format))
    return stream_handler


def get_stream_loki_handler(config: ConfigParser):
    stream_loki_handler = logging_loki.LokiHandler(
        url=f"https://{config['logging']['cloud']}/loki/api/v1/push",
        tags={
            "app": config["logging"]["app_name"],
            "uuid": config["logging"]["uiid"],
            "host": HOST,
        },
        auth=(config.getint("logging", "user"), config["logging"]["api_key_loki"]),
        version="1",
    )

    stream_loki_handler.setLevel(logging.INFO)

    return stream_loki_handler


def get_email_handler(config: ConfigParser):
    smtp_handler = logging.handlers.SMTPHandler(
        mailhost=("smtp.gmail.com", 587),
        fromaddr=config["logging"]["fromaddr"],
        toaddrs=[email for email in config["logging"]["toaddrs"].split(",") if email],
        subject=config["logging"]["subject"],
        credentials=(config["logging"]["fromaddr"], config["logging"]["api_key_gmail"]),
        secure=(),
    )

    smtp_handler.setLevel(logging.ERROR)

    return smtp_handler


def timescale(run_time: float) -> str:
    if run_time < 60:
        return f"{run_time:.2f} seconds"
    elif 60 <= run_time < 3600:
        return f"{run_time/60.:.2f} minutes"
    else:
        return f"{run_time/3600.:.2f} hours"


def task_logging(logger: logging.Logger, print_args: bool = False):
    def decorator_task_logging(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_args = dict(zip(func.__code__.co_varnames, args))
            if print_args:
                logger.info(
                    f"Task Started - {func.__name__} - Args: {func_args}, Kwargs: {kwargs}"
                )
            else:
                logger.info(f"Task Started - {func.__name__}")
            try:
                start_time = perf_counter()
                value = func(*args, **kwargs)
                end_time = perf_counter()
                run_time = end_time - start_time
                logger.info(
                    f"Task Completed - {func.__name__} - Execution time: {timescale(run_time)}",
                )
                return value
            except Exception as e:
                logger.error(
                    f"Task Uncompleted - {func.__name__} - {traceback.format_exc()}",
                )
                raise e

        return wrapper

    return decorator_task_logging
