# -*- coding: utf-8 -*-
"""
Created on: Mon August 19 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Initialiser for all logging in normits_demand
"""
# Builtins
import logging
from typing import Any, Dict

# Third Party

# Local imports
import normits_demand as nd


def get_package_logger_name():
    """Returns the name of the parent logger for this package"""
    return nd.PACKAGE_NAME


def initialise_package_logger():

    # Initiate logger
    logger = logging.getLogger(get_package_logger_name())
    logger.setLevel(logging.DEBUG)

    # Add a default console handler
    logger.addHandler(get_console_handler())


def write_instantiate_message(logger: logging.Logger,
                              instantiate_message: str,
                              ) -> None:
    """Writes an instantiation message to logger.

    Instantiation message will be output at the logging.DEBUG level,
    and will be wrapped in a line of asterisk before and after.

    Parameters
    ----------
    logger:
        The logger to write the message to.

    instantiate_message:
        The message to output on instantiation. This will be output at the
        logging.DEBUG level, and will be wrapped in a line of hyphens before
        and after.

    Returns
    -------
    None
    """
    msg = '***  %s  ***' % instantiate_message

    logger.debug('')
    logger.debug('*' * len(msg))
    logger.debug(msg)
    logger.debug('*' * len(msg))


def get_logger(logger_name: str,
               log_file_path: nd.PathLike = None,
               instantiate_msg: str = None,
               ) -> logging.Logger:
    """Creates a child logger for this package.

    Parameters
    ----------
    logger_name:
        The name of the new logger. The first name (before the dot) needs to
        be the package name, or an error will be raised.

    log_file_path:
        The path to output a logging file. If left as None, no file handler
        will be added to this logger.

    instantiate_msg:
        A message to output on instantiation. This will be output at the
        logging.DEBUG level, and will be wrapped in a line of asterisk before
        and after.

    Returns
    -------
    logger:
        A child logger of the master package logger, with a file handler
        attached, if log_file_path is set.

    Raises
    ------
    ValueError:
        if the given logger_name is not a child of this package.
    """
    check_child_logger(logger_name)

    # Create the logger
    logger = logging.getLogger(logger_name)

    # Optionally add and create a file handler
    if log_file_path is not None:
        logger.addHandler(get_file_handler(log_file=log_file_path))

    if instantiate_msg is not None:
        write_instantiate_message(logger, instantiate_msg)

    return logger


def get_custom_logger(logger_name: str,
                      console_handler: logging.StreamHandler = None,
                      file_handler: logging.FileHandler = None,
                      instantiate_msg: str = None,
                      ) -> logging.Logger:
    """Creates a child logger for this package.

    Checks to make sure the logger being created is a child of the master
    logger in the package, then creates the logger and returns.

    Parameters
    ----------
    logger_name:
        The name of the new logger. The first name (before the dot) needs to
        be the package name, or an error will be raised.

    console_handler:
        A custom logging.StreamHandler object to pass into the new logger.
        The helper function get_console_handler() can be used to get one.

    file_handler:
        A custom logging.FileHandler object to pass into the new logger.
        The helper function get_file_handler() can be used to get one.

    instantiate_msg:
        A message to output on instantiation. This will be output at the
        logging.DEBUG level, and will be wrapped in a line of asterisk before
        and after.

    Returns
    -------
    logger:
        A logger with the given handlers attached

    Raises
    ------
    ValueError:
        if the given logger_name is not a child of this package.
    """
    check_child_logger(logger_name)

    # Create the logger
    logger = logging.getLogger(logger_name)

    # Add the custom handlers if they were given
    for handler in [console_handler, file_handler]:
        if handler is not None:
            logger.addHandler(handler)

    if instantiate_msg is not None:
        write_instantiate_message(logger, instantiate_msg)

    return logger


def check_child_logger(logger_name: str) -> None:
    """Checks if the given logger name is a child of the package

    Parameters
    ----------
    logger_name:
        The name of the logger to check

    Raises
    ------
    ValueError:
        if the given logger_name is not a child of this package.
    """
    # Make sure logger is part of this package
    if logger_name.strip().split('.')[0] != nd.PACKAGE_NAME:
        raise ValueError(
            "When making a custom logger through %s the logger name needs to "
            "be a child of the master logger for this package.\n"
            "The name should be the package name followed by a dot, and the "
            "name of the logger e.g. %s.notem"
        )


def get_console_handler(ch_format: str = None,
                        datetime_format: str = None,
                        log_level: int = logging.INFO,
                        ) -> logging.StreamHandler:
    """Creates a console handles for a logger

    Parameters
    ----------
    ch_format:
        A string defining a custom formatting to use for the StreamHandler().
        Defaults to "[%(levelname)-8.8s] %(message)s".

    datetime_format:
        The datetime format to use when logging to the console.
        Defaults to "%H:%M:%S"

    log_level:
        The logging level to give to the StreamHandler.

    Returns
    -------
    console_handler:
        A logging.StreamHandler object using the format in ch_format.
    """
    if ch_format is None:
        ch_format = "[%(asctime)s - %(levelname)-8.8s] %(message)s"

    if datetime_format is None:
        datetime_format = "%H:%M:%S"

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(logging.Formatter(ch_format, datefmt=datetime_format))
    return ch


def get_file_handler(log_file: nd.PathLike,
                     fh_format: str = None,
                     datetime_format: str = None,
                     log_level: int = logging.DEBUG,
                     ) -> logging.StreamHandler:
    """Creates a console handles for a logger

    Parameters
    ----------
    log_file:
        The path to a file to output the log

    fh_format:
        A string defining a custom formatting to use for the StreamHandler().
        Defaults to
        "%(asctime)s [%(name)-40.40s] [%(levelname)-8.8s] %(message)s".

    datetime_format:
        The datetime format to use when logging to the console.
        Defaults to "%d-%m-%Y %H:%M:%S"

    log_level:
        The logging level to give to the StreamHandler.

    Returns
    -------
    console_handler:
        A logging.StreamHandler object using the format in ch_format.
    """
    # Init
    if fh_format is None:
        fh_format = "%(asctime)s [%(name)-40.40s] [%(levelname)-8.8s] %(message)s"

    if datetime_format is None:
        datetime_format = "%d-%m-%Y %H:%M:%S"

    # Create console handler
    ch = logging.FileHandler(log_file)
    ch.setLevel(log_level)
    ch.setFormatter(logging.Formatter(fh_format, datefmt=datetime_format))
    return ch


def capture_warnings(
    stream_handler: bool = True,
    stream_handler_args: Dict[str, Any] = None,
    file_handler_args: Dict[str, Any] = None
    ) -> None:
    """Capture warnings using logging.

    Runs `logging.captureWarnings(True)` to capture warnings then
    sets up custom stream and file handlers if required.

    Parameters
    ----------
    stream_handler : bool, default True
        Add stream handler to warnings logger.
    stream_handler_args : Dict[str, Any], optional
        Custom arguments for the stream handler,
        passed to `get_console_handler`.
    file_handler_args : Dict[str, Any], optional
        Custom arguments for the file handler,
        passed to `get_file_handler`.
    """
    logging.captureWarnings(True)

    warning_logger = logging.getLogger("py.warnings")

    if stream_handler or stream_handler_args is not None:
        if stream_handler_args is None:
            stream_handler_args = {}
        warning_logger.addHandler(get_console_handler(**stream_handler_args))

    if file_handler_args is not None:
        warning_logger.addHandler(get_file_handler(**file_handler_args))

