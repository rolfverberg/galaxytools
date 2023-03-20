#!/usr/bin/env python3

import sys
import argparse

def __main__():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
            description='Test')
    parser.add_argument('-l', '--log', 
            type=argparse.FileType('w'),
            default=sys.stdout,
            help='Log file')
    args = parser.parse_args()

    print('Hello world')

if __name__ == "__main__":
    __main__()

