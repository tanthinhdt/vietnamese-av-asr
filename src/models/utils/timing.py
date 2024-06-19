import time


def get_spent_time(message):
    def message_wrapper(func):
        def _wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            print(message, time.time() - start, sep=' ')
            return result
        return _wrapper
    return message_wrapper
