#!/usr/bin/env python3

import logging

import argparse
import pathlib
import sys
#import tracemalloc

#from memory_profiler import profile
#@profile
def __main__():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
            description='NeXus reader')
    parser.add_argument('-i', '--input_file',
            required=True,
            type=pathlib.Path,
            help='''Full or relative path to the input file.''')
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

    # Start memory monitoring
#    tracemalloc.start()

    # Log command line arguments
    logging.info(f'input_file = {args.input_file}')
    logging.debug(f'log = {args.log}')
    logging.debug(f'is log stdout? {args.log is sys.stdout}')
    logging.debug(f'log_level = {args.log_level}')

    # Read input file
    from nexusformat.nexus import nxload
    nxroot = nxload(args.input_file)
    print(f'nxroot:\n{nxroot.tree}')
    logging.info(f'nxroot:\n{nxroot.tree}')

    # Displaying memory usage
#    logging.info(f'Memory usage: {tracemalloc.get_traced_memory()}')
 
    # Stop memory monitoring
#    tracemalloc.stop()

    logging.info('Completed reading')


if __name__ == "__main__":
    __main__()
