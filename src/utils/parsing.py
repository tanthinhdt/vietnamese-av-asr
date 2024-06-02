import yaml
import argparse


def get_default_arg_parser(description: str = None) -> argparse.ArgumentParser:
    '''
    Get the arguments from the command line.
    Parameters
    ----------
    description : str, optional
        Description of the arguments, by default None

    Returns
    -------
    argparse.Namespace
        The arguments from the command line.
    '''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--log-path',
        type=str,
        default=None,
        help='Path to the log file',
    )
    return parser


def load_config(config_path: str) -> dict:
    '''
    Load the configuration file.

    Parameters
    ----------
    config_path : str
        The path to the configuration file.

    Returns
    -------
    dict
        The configuration file.
    '''
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
