class Processor:
    """
    This class and its children are used to perform specific tasks to process data.
    """
    def process(self, sample: dict, *args, **kwargs) -> dict:
        """
        Process sample.
        :param sample:      Sample.
        :param args:        Additional arguments.
        :param kwargs:      Additional keyword arguments.
        :return:            Processed sample.
        """
        raise NotImplementedError("Please implement this method in child classes")

    def check_output(self):
        """
        Check output.
        """
        raise NotImplementedError("Please implement this method in child classes")
