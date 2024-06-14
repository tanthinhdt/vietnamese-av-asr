import time


def get_spent_time(func):
    def _wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print()
        print(time.time() - start)
        return result
    return _wrapper
