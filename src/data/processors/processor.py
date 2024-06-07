class Processor:
    """
    This class and its children are used to perform specific tasks to process data.
    """
    def process(self, sample: dict, *args, **kwargs) -> dict:
        """
        Process sample.
        sample: 
            Dict container metadata of sample.
        args:        
            Additional arguments.
        kwargs:      
            Additional keyword arguments.
        return:
            Metadata of processed sample.
        """
        raise NotImplementedError("Please implement this method in child classes")

    def check_output(self):
        """
        Check output.
        """
        raise NotImplementedError("Please implement this method in child classes")
