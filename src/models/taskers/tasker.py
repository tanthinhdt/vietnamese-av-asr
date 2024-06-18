from abc import ABC

from typing import Any

class Tasker(ABC):
    """The abstract class for taskers."""

    def __init__(self, *args, **kwargs):
        pass

    def pre_do(self, func, *args, **kwargs):
        """
        Pre do before actually do on sample.

        func:
            Wrapped func
        args:
            Positional arguments
        kwargs:
            Keyword arguments
        return:
            Done sample
        """
        raise NotImplemented("The method is not implement")

    def do(self, sample: Any, *args, **kwargs) -> Any:
        """
        Actually do task on sample.

        sample:
            Sample need to process
        args:
            Positional arguments
        kwargs:
            Keyword arguments
        return:
            Done sample
        """
        raise NotImplemented("The method is not implement")

    def post_do(self, sample: Any, *args, **kwargs):
        """
        Post do after do on sample.

        sample:
            Sample need to process
        args:
            Positional arguments
        kwargs:
            Keyword arguments
        return:
            Done sample
        """
        raise NotImplemented("The method is not implement")
