#!/usr/bin/env python3

import logging

import argparse
import pathlib
import sys

def __main__():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
            description='Read a raw or reconstructed image')
    parser.add_argument('-i', '--input_file',
            required=True,
            type=pathlib.Path,
            help='''Full or relative path to the input file (in yaml or nxs format).''')
    parser.add_argument('--image_type',
            required=False,
            help='Image type (dark, bright, tomo_raw, tomo_reduced, or reconstructed')
    parser.add_argument('--image_index',
            required=False,
            type=int,
            help='Image index (only for raw or reduced images')
    parser.add_argument('-l', '--log',
#            type=argparse.FileType('w'),
            default=sys.stdout,
            help='Logging stream or filename')
    parser.add_argument('--log_level',
            choices=logging._nameToLevel.keys(),
            default='INFO',
            help='''Specify a preferred logging level.''')
    args = parser.parse_args()

    # Set log configuration
    # When logging to file, the stdout log level defaults to WARNING
    logging_format = '%(asctime)s : %(levelname)s - %(module)s : %(funcName)s - %(message)s'
    level = logging.getLevelName(args.log_level)
    if args.log is sys.stdout:
        logging.basicConfig(format=logging_format, level=level, force=True,
                handlers=[logging.StreamHandler()])
    else:
        if isinstance(args.log, str):
            logging.basicConfig(filename=f'{args.log}', filemode='w',
                    format=logging_format, level=level, force=True)
        elif isinstance(args.log, io.TextIOWrapper):
            logging.basicConfig(filemode='w', format=logging_format, level=level,
                    stream=args.log, force=True)
        else:
            raise(ValueError(f'Invalid argument --log: {args.log}'))
        stream_handler = logging.StreamHandler()
        logging.getLogger().addHandler(stream_handler)
        stream_handler.setLevel(logging.WARNING)
        stream_handler.setFormatter(logging.Formatter(logging_format))

    # Log command line arguments
    logging.info(f'input_file = {args.input_file}')
    logging.info(f'image_type = {args.image_type}')
    logging.info(f'image_index = {args.image_index}')
    logging.debug(f'log = {args.log}')
    logging.debug(f'is log stdout? {args.log is sys.stdout}')

    # Instantiate Tomo object
    tomo = Tomo(galaxy_flag=args.galaxy_flag)

    # Read input file
    data = tomo.read(args.input_file)
    print(f'data:\n{data}')

if __name__ == "__main__":
    __main__()

