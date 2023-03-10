#!/usr/bin/env python3

import logging

import argparse
import matplotlib.pyplot as plt
import os
#from re import compile as re_compile
import sys

def __main__():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
            description='Plot an image for the SMB schema')
    parser.add_argument('--cycle',
            required=True,
            help='''Run cycle.''')
    parser.add_argument('--station',
            required=True,
            choices=['id1a3', 'id3a'],
            help='''Beamline station.''')
    parser.add_argument('--btr',
            required=True,
            help='''BTR.''')
    parser.add_argument('--sample',
            required=True,
            help='''Sample name.''')
    parser.add_argument('--scan_number',
            default=-1,
            type=int,
            help='SPEC scan number')
    parser.add_argument('--image_index',
            default=0,
            type=int,
            help='Image index relative the first')
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
    logging.info(f'cycle = {args.cycle}')
    logging.info(f'station = {args.station}')
    logging.info(f'btr = {args.btr}')
    logging.info(f'sample = {args.sample}')
    logging.info(f'scan_number = {args.scan_number}')
    logging.info(f'image_index = {args.image_index}')
    logging.debug(f'log = {args.log}')
    logging.debug(f'is log stdout? {args.log is sys.stdout}')

    # Check input parameters
    if args.image_index < 0:
        raise ValueError(f'Invalid "image_index" parameter ({args.image_index})')

    # Check work directory
    workdir = f'/nfs/chess/{args.station}/{args.cycle}/{args.btr}/{args.sample}'
    logging.info(f'workdir = {workdir}')
    if not os.path.isdir(workdir):
        raise ValueError('Invalid work directory: {workdir}')

    # Get all available scan_numbers
    scan_numbers = [int(v) for v in os.listdir(workdir) if v.isdigit() and
            os.path.isdir(f'{workdir}/{v}')]

    if args.scan_number == -1:
        # Pick lowest scan_number with image files
        image_file = None
        for scan_number in sorted(scan_numbers):
            if 'nf' in os.listdir(f'{workdir}/{scan_number}'):
#                indexRegex = re_compile(r'\d+')
                path = f'{workdir}/{scan_number}/nf'
#                image_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and
#                        f.endswith(".tif") and indexRegex.search(f)]
                image_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
                        and f.endswith(".tif")]
                if len(image_files):
                    image_file = f'{path}/{image_files[0]}'
                    break
    else:
        # Pick requested image file
        scan_number = args.scan_number
        if not os.path.isdir(f'{workdir}/{scan_number}'):
            raise ValueError('Invalid scan_number (non-existing directory {workdir}/{scan_number})')
        path = f'{workdir}/{scan_number}/nf'
        if 'nf' not in os.listdir(f'{workdir}/{scan_number}'):
            raise ValueError('Unable to find directory {path}')
        image_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
                        and f.endswith(".tif")]
        if args.image_index >= len(image_files):
            raise ValueError('Unable to open the {args.image_index}th image file in {path}')
        image_file = f'{path}/{image_files[args.image_index]}'

    # Plot image to file
    if image_file is None:
        raise ValueError('Unable to find a valid image')
    data = plt.imread(image_file)
    title = 'image_files[0]'
    plt.figure(title)
    plt.imshow(data)
    plt.savefig('image.jpg')
    plt.close(fig=title)

if __name__ == "__main__":
    __main__()

