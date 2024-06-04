import logging

from typing import Union

def get_logger(
        name: str = None,
        level: Union[int,str] = None,
        log_path: str = None,
        is_stream: bool = False,
        format: str = None,
        fmt_type: str = None,
        **kwargs,
) -> logging.Logger:
    """
    Get logger to keep logs the flow\n
    name:
        name of logger
    level:
        nevel of logger
    log_path:
        file contains logs
    is_stream:
        log to console
    format: 
        format of message
    fmt_type:
        type of format ('normal', 'verbose')
    Return: logger
    """
    
    # refix name for logger
    if name is None:
        name = 'noname'

    if level is None:
        level = logging.DEBUG
    elif isinstance(level, str):
        level = logging._nameToLevel.get(level, logging.DEBUG)

    # create logger    
    logger = logging.getLogger(name=name)
    logger.setLevel(level=level)

    # create formatter
    if format is None:
        format = """%(asctime)s: %(name)s: %(levelname)s
            "%(filename)s", module %(module)s, line %(lineno)d in <%(funcName)s>
                --> %(message)s\
        """

    formater = logging.Formatter(fmt=format,datefmt="%m/%d/%Y, %H:%M:%S")

    # check present of handlers
    if not logger.handlers:
        # create stream handler if
        if is_stream:
            s_handler = logging.StreamHandler()
            s_handler.setLevel(level=level)
            s_handler.setFormatter(fmt=formater)
            logger.addHandler(hdlr=s_handler)
        
        # create file handler if
        if log_path is not None:
            f_handler = logging.FileHandler(
                filename=log_path,
                mode='a',
                encoding='utf-8',
            )
            f_handler.setLevel(level=level)
            f_handler.setFormatter(fmt=formater)
            logger.addHandler(hdlr=f_handler)

    return logger
    

