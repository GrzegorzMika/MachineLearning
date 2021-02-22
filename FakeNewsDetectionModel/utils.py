import logging
import time
from functools import wraps


def log_name(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        logging.info(f.__name__)
        return f(*args, **kwargs)

    return wrapper


def timer(function):

    @wraps(function)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        end = time.time() - start
        logging.info(f'{function.__name__} ran in {round(end, 2)} s')
        return result

    return wrapper
