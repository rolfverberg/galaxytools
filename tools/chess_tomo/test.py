#!/usr/bin/env python3

import logging

import sys
import argparse
import numpy as np

def __main__():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
            description='Test')
    parser.add_argument('-l', '--log', 
            type=argparse.FileType('w'),
            default=sys.stdout,
            help='Log file')
    args = parser.parse_args()

    # Set basic log configuration
    logging_format = '%(asctime)s : %(levelname)s - %(module)s : %(funcName)s - %(message)s'
    log_level = 'INFO'
    level = getattr(logging, log_level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f'Invalid log_level: {log_level}')
    logging.basicConfig(format=logging_format, level=level, force=True,
            handlers=[logging.StreamHandler()])

    logging.info(f'log = {args.log}')
    logging.info(f'is log stdout? {args.log is sys.stdout}')

if __name__ == "__main__":
    __main__()

