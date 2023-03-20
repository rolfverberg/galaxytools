#!/usr/bin/env python3

#FIX write a function that returns a list of peak indices for a given plot
#FIX use raise_error concept on more functions to optionally raise an error

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:36:22 2021

@author: rv43
"""

import logging
logger=logging.getLogger(__name__)

import os
import sys
import re
try:
    from yaml import safe_load, safe_dump
except:
    pass
try:
    import h5py
except:
    pass
import numpy as np
try:
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib import transforms
    from matplotlib.widgets import Button
except:
    pass

from ast import literal_eval
try:
    from asteval import Interpreter, get_ast_names
except:
    pass
from copy import deepcopy
try:
    from sympy import diff, simplify
except:
    pass
from time import time


def depth_list(L): return(isinstance(L, list) and max(map(depth_list, L))+1)
def depth_tuple(T): return(isinstance(T, tuple) and max(map(depth_tuple, T))+1)
def unwrap_tuple(T):
    if depth_tuple(T) > 1 and len(T) == 1:
        T = unwrap_tuple(*T)
    return(T)

def illegal_value(value, name, location=None, raise_error=False, log=True):
    if not isinstance(location, str):
        location = ''
    else:
        location = f'in {location} '
    if isinstance(name, str):
        error_msg = f'Illegal value for {name} {location}({value}, {type(value)})'
    else:
        error_msg = f'Illegal value {location}({value}, {type(value)})'
    if log:
        logger.error(error_msg)
    if raise_error:
        raise ValueError(error_msg)

def illegal_combination(value1, name1, value2, name2, location=None, raise_error=False,
        log=True):
    if not isinstance(location, str):
        location = ''
    else:
        location = f'in {location} '
    if isinstance(name1, str):
        error_msg = f'Illegal combination for {name1} and {name2} {location}'+ \
                f'({value1}, {type(value1)} and {value2}, {type(value2)})'
    else:
        error_msg = f'Illegal combination {location}'+ \
                f'({value1}, {type(value1)} and {value2}, {type(value2)})'
    if log:
        logger.error(error_msg)
    if raise_error:
        raise ValueError(error_msg)

def test_ge_gt_le_lt(ge, gt, le, lt, func, location=None, raise_error=False, log=True):
    """Check individual and mutual validity of ge, gt, le, lt qualifiers
       func: is_int or is_num to test for int or numbers
       Return: True upon success or False when mutually exlusive
    """
    if ge is None and gt is None and le is None and lt is None:
        return(True)
    if ge is not None:
        if not func(ge):
            illegal_value(ge, 'ge', location, raise_error, log) 
            return(False)
        if gt is not None:
            illegal_combination(ge, 'ge', gt, 'gt', location, raise_error, log) 
            return(False)
    elif gt is not None and not func(gt):
        illegal_value(gt, 'gt', location, raise_error, log) 
        return(False)
    if le is not None:
        if not func(le):
            illegal_value(le, 'le', location, raise_error, log) 
            return(False)
        if lt is not None:
            illegal_combination(le, 'le', lt, 'lt', location, raise_error, log) 
            return(False)
    elif lt is not None and not func(lt):
        illegal_value(lt, 'lt', location, raise_error, log) 
        return(False)
    if ge is not None:
        if le is not None and ge > le:
            illegal_combination(ge, 'ge', le, 'le', location, raise_error, log) 
            return(False)
        elif lt is not None and ge >= lt:
            illegal_combination(ge, 'ge', lt, 'lt', location, raise_error, log) 
            return(False)
    elif gt is not None:
        if le is not None and gt >= le:
            illegal_combination(gt, 'gt', le, 'le', location, raise_error, log) 
            return(False)
        elif lt is not None and gt >= lt:
            illegal_combination(gt, 'gt', lt, 'lt', location, raise_error, log) 
            return(False)
    return(True)

def range_string_ge_gt_le_lt(ge=None, gt=None, le=None, lt=None):
    """Return a range string representation matching the ge, gt, le, lt qualifiers
       Does not validate the inputs, do that as needed before calling
    """
    range_string = ''
    if ge is not None:
        if le is None and lt is None:
            range_string += f'>= {ge}'
        else:
            range_string += f'[{ge}, '
    elif gt is not None:
        if le is None and lt is None:
            range_string += f'> {gt}'
        else:
            range_string += f'({gt}, '
    if le is not None:
        if ge is None and gt is None:
            range_string += f'<= {le}'
        else:
            range_string += f'{le}]'
    elif lt is not None:
        if ge is None and gt is None:
            range_string += f'< {lt}'
        else:
            range_string += f'{lt})'
    return(range_string)

def is_int(v, ge=None, gt=None, le=None, lt=None, raise_error=False, log=True):
    """Value is an integer in range ge <= v <= le or gt < v < lt or some combination.
       Return: True if yes or False is no
    """
    return(_is_int_or_num(v, 'int', ge, gt, le, lt, raise_error, log))

def is_num(v, ge=None, gt=None, le=None, lt=None, raise_error=False, log=True):
    """Value is a number in range ge <= v <= le or gt < v < lt or some combination.
       Return: True if yes or False is no
    """
    return(_is_int_or_num(v, 'num', ge, gt, le, lt, raise_error, log))

def _is_int_or_num(v, type_str, ge=None, gt=None, le=None, lt=None, raise_error=False,
        log=True):
    if type_str == 'int':
        if not isinstance(v, int):
            illegal_value(v, 'v', '_is_int_or_num', raise_error, log) 
            return(False)
        if not test_ge_gt_le_lt(ge, gt, le, lt, is_int, '_is_int_or_num', raise_error, log):
            return(False)
    elif type_str == 'num':
        if not isinstance(v, (int, float)):
            illegal_value(v, 'v', '_is_int_or_num', raise_error, log) 
            return(False)
        if not test_ge_gt_le_lt(ge, gt, le, lt, is_num, '_is_int_or_num', raise_error, log):
            return(False)
    else:
        illegal_value(type_str, 'type_str', '_is_int_or_num', raise_error, log) 
        return(False)
    if ge is None and gt is None and le is None and lt is None:
        return(True)
    error = False
    if ge is not None and v < ge:
        error = True
        error_msg = f'Value {v} out of range: {v} !>= {ge}'
    if not error and gt is not None and v <= gt:
        error = True
        error_msg = f'Value {v} out of range: {v} !> {gt}'
    if not error and le is not None and v > le:
        error = True
        error_msg = f'Value {v} out of range: {v} !<= {le}'
    if not error and lt is not None and v >= lt:
        error = True
        error_msg = f'Value {v} out of range: {v} !< {lt}'
    if error:
        if log:
            logger.error(error_msg)
        if raise_error:
            raise ValueError(error_msg)
        return(False)
    return(True)

def is_int_pair(v, ge=None, gt=None, le=None, lt=None, raise_error=False, log=True):
    """Value is an integer pair, each in range ge <= v[i] <= le or gt < v[i] < lt or
           ge[i] <= v[i] <= le[i] or gt[i] < v[i] < lt[i] or some combination.
       Return: True if yes or False is no
    """
    return(_is_int_or_num_pair(v, 'int', ge, gt, le, lt, raise_error, log))

def is_num_pair(v, ge=None, gt=None, le=None, lt=None, raise_error=False, log=True):
    """Value is a number pair, each in range ge <= v[i] <= le or gt < v[i] < lt or
           ge[i] <= v[i] <= le[i] or gt[i] < v[i] < lt[i] or some combination.
       Return: True if yes or False is no
    """
    return(_is_int_or_num_pair(v, 'num', ge, gt, le, lt, raise_error, log))

def _is_int_or_num_pair(v, type_str, ge=None, gt=None, le=None, lt=None, raise_error=False,
        log=True):
    if type_str == 'int':
        if not (isinstance(v, (tuple, list)) and len(v) == 2 and isinstance(v[0], int) and
                isinstance(v[1], int)):
            illegal_value(v, 'v', '_is_int_or_num_pair', raise_error, log) 
            return(False)
        func = is_int
    elif type_str == 'num':
        if not (isinstance(v, (tuple, list)) and len(v) == 2 and isinstance(v[0], (int, float)) and
                isinstance(v[1], (int, float))):
            illegal_value(v, 'v', '_is_int_or_num_pair', raise_error, log) 
            return(False)
        func = is_num
    else:
        illegal_value(type_str, 'type_str', '_is_int_or_num_pair', raise_error, log)
        return(False)
    if ge is None and gt is None and le is None and lt is None:
        return(True)
    if ge is None or func(ge, log=True):
        ge = 2*[ge]
    elif not _is_int_or_num_pair(ge, type_str, raise_error=raise_error, log=log):
        return(False)
    if gt is None or func(gt, log=True):
        gt = 2*[gt]
    elif not _is_int_or_num_pair(gt, type_str, raise_error=raise_error, log=log):
        return(False)
    if le is None or func(le, log=True):
        le = 2*[le]
    elif not _is_int_or_num_pair(le, type_str, raise_error=raise_error, log=log):
        return(False)
    if lt is None or func(lt, log=True):
        lt = 2*[lt]
    elif not _is_int_or_num_pair(lt, type_str, raise_error=raise_error, log=log):
        return(False)
    if (not func(v[0], ge[0], gt[0], le[0], lt[0], raise_error, log) or
            not func(v[1], ge[1], gt[1], le[1], lt[1], raise_error, log)):
        return(False)
    return(True)

def is_int_series(l, ge=None, gt=None, le=None, lt=None, raise_error=False, log=True):
    """Value is a tuple or list of integers, each in range ge <= l[i] <= le or
       gt < l[i] < lt or some combination.
    """
    if not test_ge_gt_le_lt(ge, gt, le, lt, is_int, 'is_int_series', raise_error, log):
        return(False)
    if not isinstance(l, (tuple, list)):
        illegal_value(l, 'l', 'is_int_series', raise_error, log)
        return(False)
    if any(True if not is_int(v, ge, gt, le, lt, raise_error, log) else False for v in l):
        return(False)
    return(True)

def is_num_series(l, ge=None, gt=None, le=None, lt=None, raise_error=False, log=True):
    """Value is a tuple or list of numbers, each in range ge <= l[i] <= le or
       gt < l[i] < lt or some combination.
    """
    if not test_ge_gt_le_lt(ge, gt, le, lt, is_int, 'is_int_series', raise_error, log):
        return(False)
    if not isinstance(l, (tuple, list)):
        illegal_value(l, 'l', 'is_num_series', raise_error, log)
        return(False)
    if any(True if not is_num(v, ge, gt, le, lt, raise_error, log) else False for v in l):
        return(False)
    return(True)

def is_str_series(l, raise_error=False, log=True):
    """Value is a tuple or list of strings.
    """
    if (not isinstance(l, (tuple, list)) or
            any(True if not isinstance(s, str) else False for s in l)):
        illegal_value(l, 'l', 'is_str_series', raise_error, log)
        return(False)
    return(True)

def is_dict_series(l, raise_error=False, log=True):
    """Value is a tuple or list of dictionaries.
    """
    if (not isinstance(l, (tuple, list)) or
            any(True if not isinstance(d, dict) else False for d in l)):
        illegal_value(l, 'l', 'is_dict_series', raise_error, log)
        return(False)
    return(True)

def is_dict_nums(l, raise_error=False, log=True):
    """Value is a dictionary with single number values
    """
    if (not isinstance(l, dict) or
            any(True if not is_num(v, log=False) else False for v in l.values())):
        illegal_value(l, 'l', 'is_dict_nums', raise_error, log)
        return(False)
    return(True)

def is_dict_strings(l, raise_error=False, log=True):
    """Value is a dictionary with single string values
    """
    if (not isinstance(l, dict) or
            any(True if not isinstance(v, str) else False for v in l.values())):
        illegal_value(l, 'l', 'is_dict_strings', raise_error, log)
        return(False)
    return(True)

def is_index(v, ge=0, lt=None, raise_error=False, log=True):
    """Value is an array index in range ge <= v < lt.
       NOTE lt IS NOT included!
    """
    if isinstance(lt, int):
        if lt <= ge:
            illegal_combination(ge, 'ge', lt, 'lt', 'is_index', raise_error, log) 
            return(False)
    return(is_int(v, ge=ge, lt=lt, raise_error=raise_error, log=log))

def is_index_range(v, ge=0, le=None, lt=None, raise_error=False, log=True):
    """Value is an array index range in range ge <= v[0] <= v[1] <= le or ge <= v[0] <= v[1] < lt.
       NOTE le IS included!
    """
    if not is_int_pair(v, raise_error=raise_error, log=log):
        return(False)
    if not test_ge_gt_le_lt(ge, None, le, lt, is_int, 'is_index_range', raise_error, log):
        return(False)
    if not ge <= v[0] <= v[1] or (le is not None and v[1] > le) or (lt is not None and v[1] >= lt):
        if le is not None:
            error_msg = f'Value {v} out of range: !({ge} <= {v[0]} <= {v[1]} <= {le})'
        else:
            error_msg = f'Value {v} out of range: !({ge} <= {v[0]} <= {v[1]} < {lt})'
        if log:
            logger.error(error_msg)
        if raise_error:
            raise ValueError(error_msg)
        return(False)
    return(True)

def index_nearest(a, value):
    a = np.asarray(a)
    if a.ndim > 1:
        raise ValueError(f'Invalid array dimension for parameter a ({a.ndim}, {a})')
    # Round up for .5
    value *= 1.0+sys.float_info.epsilon
    return((int)(np.argmin(np.abs(a-value))))

def index_nearest_low(a, value):
    a = np.asarray(a)
    if a.ndim > 1:
        raise ValueError(f'Invalid array dimension for parameter a ({a.ndim}, {a})')
    index = int(np.argmin(np.abs(a-value)))
    if value < a[index] and index > 0:
        index -= 1
    return(index)

def index_nearest_upp(a, value):
    a = np.asarray(a)
    if a.ndim > 1:
        raise ValueError(f'Invalid array dimension for parameter a ({a.ndim}, {a})')
    index = int(np.argmin(np.abs(a-value)))
    if value > a[index] and index < a.size-1:
        index += 1
    return(index)

def round_to_n(x, n=1):
    if x == 0.0:
        return(0)
    else:
        return(type(x)(round(x, n-1-int(np.floor(np.log10(abs(x)))))))

def round_up_to_n(x, n=1):
    xr = round_to_n(x, n)
    if abs(x/xr) > 1.0:
        xr += np.sign(x)*10**(np.floor(np.log10(abs(x)))+1-n)
    return(type(x)(xr))

def trunc_to_n(x, n=1):
    xr = round_to_n(x, n)
    if abs(xr/x) > 1.0:
        xr -= np.sign(x)*10**(np.floor(np.log10(abs(x)))+1-n)
    return(type(x)(xr))

def almost_equal(a, b, sig_figs):
    if is_num(a) and is_num(b):
        return(abs(round_to_n(a-b, sig_figs)) < pow(10, -sig_figs+1))
    else:
        raise ValueError(f'Invalid value for a or b in almost_equal (a: {a}, {type(a)}, '+
                f'b: {b}, {type(b)})')
        return(False)

def string_to_list(s, split_on_dash=True, remove_duplicates=True, sort=True):
    """Return a list of numbers by splitting/expanding a string on any combination of
       commas, whitespaces, or dashes (when split_on_dash=True)
       e.g: '1, 3, 5-8, 12 ' -> [1, 3, 5, 6, 7, 8, 12]
    """
    if not isinstance(s, str):
        illegal_value(s, location='string_to_list') 
        return(None)
    if not len(s):
        return([])
    try:
        ll = [x for x in re.split('\s+,\s+|\s+,|,\s+|\s+|,', s.strip())]
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return(None)
    if split_on_dash:
        try:
            l = []
            for l1 in ll:
                l2 = [literal_eval(x) for x in re.split('\s+-\s+|\s+-|-\s+|\s+|-', l1)]
                if len(l2) == 1:
                    l += l2
                elif len(l2) == 2 and l2[1] > l2[0]:
                    l += [i for i in range(l2[0], l2[1]+1)]
                else:
                    raise ValueError
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return(None)
    else:
        l = [literal_eval(x) for x in ll]
    if remove_duplicates:
        l = list(dict.fromkeys(l))
    if sort:
        l = sorted(l)
    return(l)

def get_trailing_int(string):
    indexRegex = re.compile(r'\d+$')
    mo = indexRegex.search(string)
    if mo is None:
        return(None)
    else:
        return(int(mo.group()))

def input_int(s=None, ge=None, gt=None, le=None, lt=None, default=None, inset=None,
        raise_error=False, log=True):
    return(_input_int_or_num('int', s, ge, gt, le, lt, default, inset, raise_error, log))

def input_num(s=None, ge=None, gt=None, le=None, lt=None, default=None, raise_error=False,
        log=True):
    return(_input_int_or_num('num', s, ge, gt, le, lt, default, None, raise_error,log))

def _input_int_or_num(type_str, s=None, ge=None, gt=None, le=None, lt=None, default=None,
        inset=None, raise_error=False, log=True):
    if type_str == 'int':
        if not test_ge_gt_le_lt(ge, gt, le, lt, is_int, '_input_int_or_num', raise_error, log):
            return(None)
    elif type_str == 'num':
        if not test_ge_gt_le_lt(ge, gt, le, lt, is_num, '_input_int_or_num', raise_error, log):
            return(None)
    else:
        illegal_value(type_str, 'type_str', '_input_int_or_num', raise_error, log)
        return(None)
    if default is not None:
        if not _is_int_or_num(default, type_str, raise_error=raise_error, log=log):
            return(None)
        if ge is not None and default < ge:
            illegal_combination(ge, 'ge', default, 'default', '_input_int_or_num', raise_error,
                log)
            return(None)
        if gt is not None and default <= gt:
            illegal_combination(gt, 'gt', default, 'default', '_input_int_or_num', raise_error,
                log)
            return(None)
        if le is not None and default > le:
            illegal_combination(le, 'le', default, 'default', '_input_int_or_num', raise_error,
                log)
            return(None)
        if lt is not None and default >= lt:
            illegal_combination(lt, 'lt', default, 'default', '_input_int_or_num', raise_error,
                log)
            return(None)
        default_string = f' [{default}]'
    else:
        default_string = ''
    if inset is not None:
        if (not isinstance(inset, (tuple, list)) or any(True if not isinstance(i, int) else
                False for i in inset)):
            illegal_value(inset, 'inset', '_input_int_or_num', raise_error, log) 
            return(None)
    v_range = f'{range_string_ge_gt_le_lt(ge, gt, le, lt)}'
    if len(v_range):
        v_range = f' {v_range}'
    if s is None:
        if type_str == 'int':
            print(f'Enter an integer{v_range}{default_string}: ')
        else:
            print(f'Enter a number{v_range}{default_string}: ')
    else:
        print(f'{s}{v_range}{default_string}: ')
    try:
        i = input()
        if isinstance(i, str) and not len(i):
            v = default
            print(f'{v}')
        else:
            v = literal_eval(i)
        if inset and v not in inset:
           raise ValueError(f'{v} not part of the set {inset}')
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        v = None
    except:
        if log:
            logger.error('Unexpected error')
        if raise_error:
            raise ValueError('Unexpected error')
    if not _is_int_or_num(v, type_str, ge, gt, le, lt):
        v = _input_int_or_num(type_str, s, ge, gt, le, lt, default, inset, raise_error, log)
    return(v)

def input_int_list(s=None, ge=None, le=None, split_on_dash=True, remove_duplicates=True,
        sort=True, raise_error=False, log=True):
    """Prompt the user to input a list of interger and split the entered string on any combination
       of commas, whitespaces, or dashes (when split_on_dash is True)
       e.g: '1 3,5-8 , 12 ' -> [1, 3, 5, 6, 7, 8, 12]
       remove_duplicates: removes duplicates if True (may also change the order)
       sort: sort in ascending order if True
       return None upon an illegal input
    """
    return(_input_int_or_num_list('int', s, ge, le, split_on_dash, remove_duplicates, sort,
        raise_error, log))

def input_num_list(s=None, ge=None, le=None, remove_duplicates=True, sort=True, raise_error=False,
        log=True):
    """Prompt the user to input a list of numbers and split the entered string on any combination
       of commas or whitespaces
       e.g: '1.0, 3, 5.8, 12 ' -> [1.0, 3.0, 5.8, 12.0]
       remove_duplicates: removes duplicates if True (may also change the order)
       sort: sort in ascending order if True
       return None upon an illegal input
    """
    return(_input_int_or_num_list('num', s, ge, le, False, remove_duplicates, sort, raise_error,
        log))

def _input_int_or_num_list(type_str, s=None, ge=None, le=None, split_on_dash=True,
        remove_duplicates=True, sort=True, raise_error=False, log=True):
    #FIX do we want a limit on max dimension?
    if type_str == 'int':
        if not test_ge_gt_le_lt(ge, None, le, None, is_int, 'input_int_or_num_list', raise_error,
                log):
            return(None)
    elif type_str == 'num':
        if not test_ge_gt_le_lt(ge, None, le, None, is_num, 'input_int_or_num_list', raise_error,
                log):
            return(None)
    else:
        illegal_value(type_str, 'type_str', '_input_int_or_num_list')
        return(None)
    v_range = f'{range_string_ge_gt_le_lt(ge=ge, le=le)}'
    if len(v_range):
        v_range = f' (each value in {v_range})'
    if s is None:
        print(f'Enter a series of integers{v_range}: ')
    else:
        print(f'{s}{v_range}: ')
    try:
        l = string_to_list(input(), split_on_dash, remove_duplicates, sort)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        l = None
    except:
        print('Unexpected error')
        raise
    if (not isinstance(l, list) or
            any(True if not _is_int_or_num(v, type_str, ge=ge, le=le) else False for v in l)):
        if split_on_dash:
            print('Invalid input: enter a valid set of dash/comma/whitespace separated integers '+
                    'e.g. 1 3,5-8 , 12')
        else:
            print('Invalid input: enter a valid set of comma/whitespace separated integers '+
                    'e.g. 1 3,5 8 , 12')
        l = _input_int_or_num_list(type_str, s, ge, le, split_on_dash, remove_duplicates, sort,
            raise_error, log)
    return(l)

def input_yesno(s=None, default=None):
    if default is not None:
        if not isinstance(default, str):
            illegal_value(default, 'default', 'input_yesno') 
            return(None)
        if default.lower() in 'yes':
            default = 'y'
        elif default.lower() in 'no':
            default = 'n'
        else:
            illegal_value(default, 'default', 'input_yesno') 
            return(None)
        default_string = f' [{default}]'
    else:
        default_string = ''
    if s is None:
        print(f'Enter yes or no{default_string}: ')
    else:
        print(f'{s}{default_string}: ')
    i = input()
    if isinstance(i, str) and not len(i):
        i = default
        print(f'{i}')
    if i is not None and i.lower() in 'yes':
        v = True
    elif i is not None and i.lower() in 'no':
        v = False
    else:
        print('Invalid input, enter yes or no')
        v = input_yesno(s, default)
    return(v)

def input_menu(items, default=None, header=None):
    if not isinstance(items, (tuple, list)) or any(True if not isinstance(i, str) else False
            for i in items):
        illegal_value(items, 'items', 'input_menu') 
        return(None)
    if default is not None:
        if not (isinstance(default, str) and default in items):
            logger.error(f'Invalid value for default ({default}), must be in {items}') 
            return(None)
        default_string = f' [{items.index(default)+1}]'
    else:
        default_string = ''
    if header is None:
        print(f'Choose one of the following items (1, {len(items)}){default_string}:')
    else:
        print(f'{header} (1, {len(items)}){default_string}:')
    for i, choice in enumerate(items):
        print(f'  {i+1}: {choice}')
    try:
        choice  = input()
        if isinstance(choice, str) and not len(choice):
            choice = items.index(default)
            print(f'{choice+1}')
        else:
            choice = literal_eval(choice)
            if isinstance(choice, int) and 1 <= choice <= len(items):
                choice -= 1
            else:
                raise ValueError
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        choice = None
    except:
        print('Unexpected error')
        raise
    if choice is None:
        print(f'Invalid choice, enter a number between 1 and {len(items)}')
        choice = input_menu(items, default)
    return(choice)

def assert_no_duplicates_in_list_of_dicts(l: list, raise_error=False) -> list:
    if not isinstance(l, list):
        illegal_value(l, 'l', 'assert_no_duplicates_in_list_of_dicts', raise_error)
        return(None)
    if any(True if not isinstance(d, dict) else False for d in l):
        illegal_value(l, 'l', 'assert_no_duplicates_in_list_of_dicts', raise_error)
        return(None)
    if len(l) != len([dict(t) for t in {tuple(sorted(d.items())) for d in l}]):
        if raise_error:
            raise ValueError(f'Duplicate items found in {l}')
        else:
            logger.error(f'Duplicate items found in {l}')
        return(None)
    else:
        return(l)

def assert_no_duplicate_key_in_list_of_dicts(l: list, key: str, raise_error=False) -> list:
    if not isinstance(key, str):
        illegal_value(key, 'key', 'assert_no_duplicate_key_in_list_of_dicts', raise_error)
        return(None)
    if not isinstance(l, list):
        illegal_value(l, 'l', 'assert_no_duplicate_key_in_list_of_dicts', raise_error)
        return(None)
    if any(True if not isinstance(d, dict) else False for d in l):
        illegal_value(l, 'l', 'assert_no_duplicates_in_list_of_dicts', raise_error)
        return(None)
    keys = [d.get(key, None) for d in l]
    if None in keys or len(set(keys)) != len(l):
        if raise_error:
            raise ValueError(f'Duplicate or missing key ({key}) found in {l}')
        else:
            logger.error(f'Duplicate or missing key ({key}) found in {l}')
        return(None)
    else:
        return(l)

def assert_no_duplicate_attr_in_list_of_objs(l: list, attr: str, raise_error=False) -> list:
    if not isinstance(attr, str):
        illegal_value(attr, 'attr', 'assert_no_duplicate_attr_in_list_of_objs', raise_error)
        return(None)
    if not isinstance(l, list):
        illegal_value(l, 'l', 'assert_no_duplicate_key_in_list_of_objs', raise_error)
        return(None)
    attrs = [getattr(obj, attr, None) for obj in l]
    if None in attrs or len(set(attrs)) != len(l):
        if raise_error:
            raise ValueError(f'Duplicate or missing attr ({attr}) found in {l}')
        else:
            logger.error(f'Duplicate or missing attr ({attr}) found in {l}')
        return(None)
    else:
        return(l)

def file_exists_and_readable(path):
    if not os.path.isfile(path):
        raise ValueError(f'{path} is not a valid file')
    elif not os.access(path, os.R_OK):
        raise ValueError(f'{path} is not accessible for reading')
    else:
        return(path)

def create_mask(x, bounds=None, exclude_bounds=False, current_mask=None):
    # bounds is a pair of number in the same units a x
    if not isinstance(x, (tuple, list, np.ndarray)) or not len(x):
        logger.warning(f'Invalid input array ({x}, {type(x)})')
        return(None)
    if bounds is not None and not is_num_pair(bounds):
        logger.warning(f'Invalid bounds parameter ({bounds} {type(bounds)}, input ignored')
        bounds = None
    if bounds is not None:
        if exclude_bounds:
            mask = np.logical_or(x < min(bounds), x > max(bounds))
        else:
            mask = np.logical_and(x > min(bounds), x < max(bounds))
    else:
        mask = np.ones(len(x), dtype=bool)
    if current_mask is not None:
        if not isinstance(current_mask, (tuple, list, np.ndarray)) or len(current_mask) != len(x):
            logger.warning(f'Invalid current_mask ({current_mask}, {type(current_mask)}), '+
                    'input ignored')
        else:
            mask = np.logical_or(mask, current_mask)
    if not True in mask:
        logger.warning('Entire data array is masked')
    return(mask)

def eval_expr(name, expr, expr_variables, user_variables=None, max_depth=10, raise_error=False,
        log=True, **kwargs):
    """Evaluate an expression of expressions
    """
    if not isinstance(name, str):
        illegal_value(name, 'name', 'eval_expr', raise_error, log)
        return(None)
    if not isinstance(expr, str):
        illegal_value(expr, 'expr', 'eval_expr', raise_error, log)
        return(None)
    if not is_dict_strings(expr_variables, log=False):
        illegal_value(expr_variables, 'expr_variables', 'eval_expr', raise_error, log)
        return(None)
    if user_variables is not None and not is_dict_nums(user_variables, log=False):
        illegal_value(user_variables, 'user_variables', 'eval_expr', raise_error, log)
        return(None)
    if not is_int(max_depth, gt=1, log=False):
        illegal_value(max_depth, 'max_depth', 'eval_expr', raise_error, log)
        return(None)
    if not isinstance(raise_error, bool):
        illegal_value(raise_error, 'raise_error', 'eval_expr', raise_error, log)
        return(None)
    if not isinstance(log, bool):
        illegal_value(log, 'log', 'eval_expr', raise_error, log)
        return(None)
#    print(f'\nEvaluate the full expression for {expr}')
    if 'chain' in kwargs:
        chain = kwargs.pop('chain')
        if not is_str_series(chain):
            illegal_value(chain, 'chain', 'eval_expr', raise_error, log)
            return(None)
    else:
        chain = []
    if len(chain) > max_depth:
        error_msg = 'Exceeded maximum depth ({max_depth}) in eval_expr'
        if log:
            logger.error(error_msg)
        if raise_error:
            raise ValueError(error_msg)
        return(None)
    if name not in chain:
        chain.append(name)
#    print(f'start: chain = {chain}')
    if 'ast' in kwargs:
        ast = kwargs.pop('ast')
    else:
        ast = Interpreter()
    if user_variables is not None:
        ast.symtable.update(user_variables)
    chain_vars = [var for var in get_ast_names(ast.parse(expr))
            if var in expr_variables and var not in ast.symtable]
#    print(f'chain_vars: {chain_vars}')
    save_chain = chain.copy()
    for var in chain_vars:
#        print(f'\n\tname = {name}, var = {var}:\n\t\t{expr_variables[var]}')
#        print(f'\tchain = {chain}')
        if var in chain:
            error_msg = f'Circular variable {var} in eval_expr'
            if log:
               logger.error(error_msg)
            if raise_error:
               raise ValueError(error_msg)
            return(None)
#        print(f'\tknown symbols:\n\t\t{ast.user_defined_symbols()}\n')
        if var in ast.user_defined_symbols():
            val = ast.symtable[var]
        else:
            #val = eval_expr(var, expr_variables[var], expr_variables, user_variables=user_variables,
            val = eval_expr(var, expr_variables[var], expr_variables, max_depth=max_depth,
                    raise_error=raise_error, log=log, chain=chain, ast=ast)
            if val is None:
                return(None)
            ast.symtable[var] = val
#        print(f'\tval = {val}')
#        print(f'\t{var} = {ast.symtable[var]}')
        chain = save_chain.copy()
#        print(f'\treset loop for {var}: chain = {chain}')
    val = ast.eval(expr)
#    print(f'return val for {expr} = {val}\n')
    return(val)

def full_gradient(expr, x, expr_name=None, expr_variables=None, valid_variables=None, max_depth=10,
        raise_error=False, log=True, **kwargs):
    """Compute the full gradient dexpr/dx
    """
    if not isinstance(x, str):
        illegal_value(x, 'x', 'full_gradient', raise_error, log)
        return(None)
    if expr_name is not None and not isinstance(expr_name, str):
        illegal_value(expr_name, 'expr_name', 'eval_expr', raise_error, log)
        return(None)
    if expr_variables is not None and not is_dict_strings(expr_variables, log=False):
        illegal_value(expr_variables, 'expr_variables', 'full_gradient', raise_error, log)
        return(None)
    if valid_variables is not None and not is_str_series(valid_variables, log=False):
        illegal_value(valid_variables, 'valid_variables', 'full_gradient', raise_error, log)
    if not is_int(max_depth, gt=1, log=False):
        illegal_value(max_depth, 'max_depth', 'eval_expr', raise_error, log)
        return(None)
    if not isinstance(raise_error, bool):
        illegal_value(raise_error, 'raise_error', 'eval_expr', raise_error, log)
        return(None)
    if not isinstance(log, bool):
        illegal_value(log, 'log', 'eval_expr', raise_error, log)
        return(None)
#    print(f'\nGet full gradient of {expr_name} = {expr} with respect to {x}')
    if expr_name is not None and expr_name == x:
        return(1.0)
    if 'chain' in kwargs:
        chain = kwargs.pop('chain')
        if not is_str_series(chain):
            illegal_value(chain, 'chain', 'eval_expr', raise_error, log)
            return(None)
    else:
        chain = []
    if len(chain) > max_depth:
        error_msg = 'Exceeded maximum depth ({max_depth}) in eval_expr'
        if log:
            logger.error(error_msg)
        if raise_error:
            raise ValueError(error_msg)
        return(None)
    if expr_name is not None and expr_name not in chain:
        chain.append(expr_name)
#    print(f'start ({x}): chain = {chain}')
    ast = Interpreter()
    if expr_variables is None:
        chain_vars = []
    else:
        chain_vars = [var for var in get_ast_names(ast.parse(f'{expr}'))
                if var in expr_variables and var != x and var not in ast.symtable]
#    print(f'chain_vars: {chain_vars}')
    if valid_variables is not None:
        unknown_vars = [var for var in chain_vars if var not in valid_variables]
        if len(unknown_vars):
            error_msg = f'Unknown variable {unknown_vars} in {expr}'
            if log:
               logger.error(error_msg)
            if raise_error:
               raise ValueError(error_msg)
            return(None)
    dexpr_dx = diff(expr, x)
#    print(f'direct gradient: d({expr})/d({x}) = {dexpr_dx} ({type(dexpr_dx)})')
    save_chain = chain.copy()
    for var in chain_vars:
#        print(f'\n\texpr_name = {expr_name}, var = {var}:\n\t\t{expr}')
#        print(f'\tchain = {chain}')
        if var in chain:
            error_msg = f'Circular variable {var} in full_gradient'
            if log:
               logger.error(error_msg)
            if raise_error:
               raise ValueError(error_msg)
            return(None)
        dexpr_dvar = diff(expr, var)
#        print(f'\td({expr})/d({var}) = {dexpr_dvar}')
        if dexpr_dvar:
            dvar_dx = full_gradient(expr_variables[var], x, expr_name=var,
                    expr_variables=expr_variables, valid_variables=valid_variables,
                    max_depth=max_depth, raise_error=raise_error, log=log, chain=chain)
#            print(f'\t\td({var})/d({x}) = {dvar_dx}')
            if dvar_dx:
                dexpr_dx = f'{dexpr_dx}+({dexpr_dvar})*({dvar_dx})'
#            print(f'\t\t2: chain = {chain}')
        chain = save_chain.copy()
#        print(f'\treset loop for {var}: chain = {chain}')
#    print(f'full gradient: d({expr})/d({x}) = {dexpr_dx} ({type(dexpr_dx)})')
#    print(f'reset end: chain = {chain}\n\n')
    return(simplify(dexpr_dx))

def bounds_from_mask(mask, return_include_bounds:bool=True):
    bounds = []
    for i, m in enumerate(mask):
        if m == return_include_bounds:
            if len(bounds) == 0 or type(bounds[-1]) == tuple:
                bounds.append(i)
        else:
            if len(bounds) > 0 and isinstance(bounds[-1], int):
                bounds[-1] = (bounds[-1], i-1)
    if len(bounds) > 0 and isinstance(bounds[-1], int):
        bounds[-1] = (bounds[-1], mask.size-1)
    return(bounds)

def draw_mask_1d(ydata, xdata=None, current_index_ranges=None, current_mask=None,
        select_mask=True, num_index_ranges_max=None, title=None, legend=None, test_mode=False):
    #FIX make color blind friendly
    def draw_selections(ax, current_include, current_exclude, selected_index_ranges):
        ax.clear()
        ax.set_title(title)
        ax.legend([legend])
        ax.plot(xdata, ydata, 'k')
        for (low, upp) in current_include:
            xlow = 0.5*(xdata[max(0, low-1)]+xdata[low])
            xupp = 0.5*(xdata[upp]+xdata[min(num_data-1, upp+1)])
            ax.axvspan(xlow, xupp, facecolor='green', alpha=0.5)
        for (low, upp) in current_exclude:
            xlow = 0.5*(xdata[max(0, low-1)]+xdata[low])
            xupp = 0.5*(xdata[upp]+xdata[min(num_data-1, upp+1)])
            ax.axvspan(xlow, xupp, facecolor='red', alpha=0.5)
        for (low, upp) in selected_index_ranges:
            xlow = 0.5*(xdata[max(0, low-1)]+xdata[low])
            xupp = 0.5*(xdata[upp]+xdata[min(num_data-1, upp+1)])
            ax.axvspan(xlow, xupp, facecolor=selection_color, alpha=0.5)
        ax.get_figure().canvas.draw()

    def onclick(event):
        if event.inaxes in [fig.axes[0]]:
            selected_index_ranges.append(index_nearest_upp(xdata, event.xdata))

    def onrelease(event):
        if len(selected_index_ranges) > 0:
            if isinstance(selected_index_ranges[-1], int):
                if event.inaxes in [fig.axes[0]]:
                    event.xdata = index_nearest_low(xdata, event.xdata)
                    if selected_index_ranges[-1] <= event.xdata:
                        selected_index_ranges[-1] = (selected_index_ranges[-1], event.xdata)
                    else:
                        selected_index_ranges[-1] = (event.xdata, selected_index_ranges[-1])
                    draw_selections(event.inaxes, current_include, current_exclude, selected_index_ranges)
                else:
                    selected_index_ranges.pop(-1)

    def confirm_selection(event):
        plt.close()
    
    def clear_last_selection(event):
        if len(selected_index_ranges):
            selected_index_ranges.pop(-1)
        else:
            while len(current_include):
                current_include.pop()
            while len(current_exclude):
                current_exclude.pop()
            selected_mask.fill(False)
        draw_selections(ax, current_include, current_exclude, selected_index_ranges)

    def update_mask(mask, selected_index_ranges, unselected_index_ranges):
        for (low, upp) in selected_index_ranges:
            selected_mask = np.logical_and(xdata >= xdata[low], xdata <= xdata[upp])
            mask = np.logical_or(mask, selected_mask)
        for (low, upp) in unselected_index_ranges:
            unselected_mask = np.logical_and(xdata >= xdata[low], xdata <= xdata[upp])
            mask[unselected_mask] = False
        return(mask)

    def update_index_ranges(mask):
        # Update the currently included index ranges (where mask is True)
        current_include = []
        for i, m in enumerate(mask):
            if m == True:
                if len(current_include) == 0 or type(current_include[-1]) == tuple:
                    current_include.append(i)
            else:
                if len(current_include) > 0 and isinstance(current_include[-1], int):
                    current_include[-1] = (current_include[-1], i-1)
        if len(current_include) > 0 and isinstance(current_include[-1], int):
            current_include[-1] = (current_include[-1], num_data-1)
        return(current_include)

    # Check inputs
    ydata = np.asarray(ydata)
    if ydata.ndim > 1:
        logger.warning(f'Invalid ydata dimension ({ydata.ndim})')
        return(None, None)
    num_data = ydata.size
    if xdata is None:
        xdata = np.arange(num_data)
    else:
        xdata = np.asarray(xdata, dtype=np.float64)
        if xdata.ndim > 1 or xdata.size != num_data:
            logger.warning(f'Invalid xdata shape ({xdata.shape})')
            return(None, None)
        if not np.all(xdata[:-1] < xdata[1:]):
            logger.warning('Invalid xdata: must be monotonically increasing')
            return(None, None)
    if current_index_ranges is not None:
        if not isinstance(current_index_ranges, (tuple, list)):
            logger.warning('Invalid current_index_ranges parameter ({current_index_ranges}, '+
                    f'{type(current_index_ranges)})')
            return(None, None)
    if not isinstance(select_mask, bool):
        logger.warning('Invalid select_mask parameter ({select_mask}, {type(select_mask)})')
        return(None, None)
    if num_index_ranges_max is not None:
        logger.warning('num_index_ranges_max input not yet implemented in draw_mask_1d')
    if title is None:
        title = 'select ranges of data'
    elif not isinstance(title, str):
        illegal(title, 'title')
        title = ''
    if legend is None and not isinstance(title, str):
        illegal(legend, 'legend')
        legend = None

    if select_mask:
        title = f'Click and drag to {title} you wish to include'
        selection_color = 'green'
    else:
        title = f'Click and drag to {title} you wish to exclude'
        selection_color = 'red'

    # Set initial selected mask and the selected/unselected index ranges as needed
    selected_index_ranges = []
    unselected_index_ranges = []
    selected_mask = np.full(xdata.shape, False, dtype=bool)
    if current_index_ranges is None:
        if current_mask is None:
            if not select_mask:
                selected_index_ranges = [(0, num_data-1)]
                selected_mask = np.full(xdata.shape, True, dtype=bool)
        else:
            selected_mask = np.copy(np.asarray(current_mask, dtype=bool))
    if current_index_ranges is not None and len(current_index_ranges):
        current_index_ranges = sorted([(low, upp) for (low, upp) in current_index_ranges])
        for (low, upp) in current_index_ranges:
            if low > upp or low >= num_data or upp < 0:
                continue
            if low < 0:
                low = 0
            if upp >= num_data:
                upp = num_data-1
            selected_index_ranges.append((low, upp))
        selected_mask = update_mask(selected_mask, selected_index_ranges, unselected_index_ranges)
    if current_index_ranges is not None and current_mask is not None:
        selected_mask = np.logical_and(current_mask, selected_mask)
    if current_mask is not None:
        selected_index_ranges = update_index_ranges(selected_mask)

    # Set up range selections for display
    current_include = selected_index_ranges
    current_exclude = []
    selected_index_ranges = []
    if not len(current_include):
        if select_mask:
            current_exclude = [(0, num_data-1)]
        else:
            current_include = [(0, num_data-1)]
    else:
        if current_include[0][0] > 0:
            current_exclude.append((0, current_include[0][0]-1))
        for i in range(1, len(current_include)):
            current_exclude.append((current_include[i-1][1]+1, current_include[i][0]-1))
        if current_include[-1][1] < num_data-1:
            current_exclude.append((current_include[-1][1]+1, num_data-1))

    if not test_mode:

        # Set up matplotlib figure
        plt.close('all')
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        draw_selections(ax, current_include, current_exclude, selected_index_ranges)

        # Set up event handling for click-and-drag range selection
        cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
        cid_release = fig.canvas.mpl_connect('button_release_event', onrelease)

        # Set up confirm / clear range selection buttons
        confirm_b = Button(plt.axes([0.75, 0.05, 0.15, 0.075]), 'Confirm')
        clear_b = Button(plt.axes([0.59, 0.05, 0.15, 0.075]), 'Clear')
        cid_confirm = confirm_b.on_clicked(confirm_selection)
        cid_clear = clear_b.on_clicked(clear_last_selection)

        # Show figure
        plt.show(block=True)

        # Disconnect callbacks when figure is closed
        fig.canvas.mpl_disconnect(cid_click)
        fig.canvas.mpl_disconnect(cid_release)
        confirm_b.disconnect(cid_confirm)
        clear_b.disconnect(cid_clear)

    # Swap selection depending on select_mask
    if not select_mask:
        selected_index_ranges, unselected_index_ranges = unselected_index_ranges, \
                selected_index_ranges

    # Update the mask with the currently selected/unselected x-ranges
    selected_mask = update_mask(selected_mask, selected_index_ranges, unselected_index_ranges)

    # Update the currently included index ranges (where mask is True)
    current_include = update_index_ranges(selected_mask)
    
    return(selected_mask, current_include)

def select_peaks(ydata:np.ndarray, x_values:np.ndarray=None, x_mask:np.ndarray=None,
                 peak_x_values:np.ndarray=np.array([]), peak_x_indices:np.ndarray=np.array([]),
                 return_peak_x_values:bool=False, return_peak_x_indices:bool=False,
                 return_peak_input_indices:bool=False, return_sorted:bool=False,
                 title:str=None, xlabel:str=None, ylabel:str=None) -> list :

    # Check arguments
    if (len(peak_x_values) > 0 or return_peak_x_values) and not len(x_values) > 0:
        raise RuntimeError('Cannot use peak_x_values or return_peak_x_values without x_values')
    if not ((len(peak_x_values) > 0) ^ (len(peak_x_indices) > 0)):
        raise RuntimeError('Use exactly one of peak_x_values or peak_x_indices')
    return_format_iter = iter((return_peak_x_values, return_peak_x_indices, return_peak_input_indices))
    if not (any(return_format_iter) and not any(return_format_iter)):
        raise RuntimeError('Exactly one of return_peak_x_values, return_peak_x_indices, or '+
                'return_peak_input_indices must be True')

    EXCLUDE_PEAK_PROPERTIES = {'color': 'black', 'linestyle': '--','linewidth': 1,
                               'marker': 10, 'markersize': 5, 'fillstyle': 'none'}
    INCLUDE_PEAK_PROPERTIES = {'color': 'green', 'linestyle': '-', 'linewidth': 2,
                               'marker': 10, 'markersize': 10, 'fillstyle': 'full'}
    MASKED_PEAK_PROPERTIES = {'color': 'gray', 'linestyle': ':', 'linewidth': 1}

    # Setup reference data & plot
    x_indices = np.arange(len(ydata))
    if x_values is None:
        x_values = x_indices
    if x_mask is None:
        x_mask = np.full(x_values.shape, True, dtype=bool)
    fig, ax = plt.subplots()
    handles = ax.plot(x_values, ydata, label='Reference data')
    handles.append(mlines.Line2D([], [], label='Excluded / unselected HKL', **EXCLUDE_PEAK_PROPERTIES))
    handles.append(mlines.Line2D([], [], label='Included / selected HKL', **INCLUDE_PEAK_PROPERTIES))
    handles.append(mlines.Line2D([], [], label='HKL in masked region (unselectable)', **MASKED_PEAK_PROPERTIES))
    ax.legend(handles=handles, loc='upper right')
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)


    # Plot vertical line at each peak
    value_to_index = lambda x_value: int(np.argmin(abs(x_values - x_value)))
    if len(peak_x_indices) > 0:
        peak_x_values = x_values[peak_x_indices]
    else:
        peak_x_indices = np.array(list(map(value_to_index, peak_x_values)))
    peak_vlines = []
    for loc in peak_x_values:
        nearest_index = value_to_index(loc)
        if nearest_index in x_indices[x_mask]:
            peak_vline = ax.axvline(loc, **EXCLUDE_PEAK_PROPERTIES)
            peak_vline.set_picker(5)
        else:
            peak_vline = ax.axvline(loc, **MASKED_PEAK_PROPERTIES)
        peak_vlines.append(peak_vline)

    # Indicate masked regions by gray-ing out the axes facecolor
    mask_exclude_bounds = bounds_from_mask(x_mask, return_include_bounds=False)
    for (low, upp) in mask_exclude_bounds:
        xlow = x_values[low]
        xupp = x_values[upp]
        ax.axvspan(xlow, xupp, facecolor='gray', alpha=0.5)

    # Setup peak picking
    selected_peak_input_indices = []
    def onpick(event):
        try:
            peak_index = peak_vlines.index(event.artist)
        except:
            pass
        else:
            peak_vline = event.artist
            if peak_index in selected_peak_input_indices:
                peak_vline.set(**EXCLUDE_PEAK_PROPERTIES)
                selected_peak_input_indices.remove(peak_index)
            else:
                peak_vline.set(**INCLUDE_PEAK_PROPERTIES)
                selected_peak_input_indices.append(peak_index)
            plt.draw()
    cid_pick_peak = fig.canvas.mpl_connect('pick_event', onpick)

    # Setup "Confirm" button
    def confirm_selection(event):
        plt.close()
    plt.subplots_adjust(bottom=0.2)
    confirm_b = Button(plt.axes([0.75, 0.05, 0.15, 0.075]), 'Confirm')
    cid_confirm = confirm_b.on_clicked(confirm_selection)

    # Show figure for user interaction
    plt.show()

    # Disconnect callbacks when figure is closed
    fig.canvas.mpl_disconnect(cid_pick_peak)
    confirm_b.disconnect(cid_confirm)

    if return_peak_input_indices:
        selected_peaks = np.array(selected_peak_input_indices)
    if return_peak_x_values:
        selected_peaks = peak_x_values[selected_peak_input_indices]
    if return_peak_x_indices:
        selected_peaks = peak_x_indices[selected_peak_input_indices]

    if return_sorted:
        selected_peaks.sort()

    return(selected_peaks)

def find_image_files(path, filetype, name=None):
    if isinstance(name, str):
        name = f'{name.strip()} '
    else:
        name = ''
    # Find available index range
    if filetype == 'tif':
        if not isinstance(path, str) or not os.path.isdir(path):
            illegal_value(path, 'path', 'find_image_files')
            return(-1, 0, [])
        indexRegex = re.compile(r'\d+')
        # At this point only tiffs
        files = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and
                f.endswith('.tif') and indexRegex.search(f)])
        num_img = len(files)
        if num_img < 1:
            logger.warning(f'No available {name}files')
            return(-1, 0, [])
        first_index = indexRegex.search(files[0]).group()
        last_index = indexRegex.search(files[-1]).group()
        if first_index is None or last_index is None:
            logger.error(f'Unable to find correctly indexed {name}images')
            return(-1, 0, [])
        first_index = int(first_index)
        last_index = int(last_index)
        if num_img != last_index-first_index+1:
            logger.error(f'Non-consecutive set of indices for {name}images')
            return(-1, 0, [])
        paths = [os.path.join(path, f) for f in files]
    elif filetype == 'h5':
        if not isinstance(path, str) or not os.path.isfile(path):
            illegal_value(path, 'path', 'find_image_files')
            return(-1, 0, [])
        # At this point only h5 in alamo2 detector style
        first_index = 0
        with h5py.File(path, 'r') as f:
            num_img = f['entry/instrument/detector/data'].shape[0]
            last_index = num_img-1
        paths = [path]
    else:
        illegal_value(filetype, 'filetype', 'find_image_files')
        return(-1, 0, [])
    logger.info(f'Number of available {name}images: {num_img}')
    logger.info(f'Index range of available {name}images: [{first_index}, '+
            f'{last_index}]')

    return(first_index, num_img, paths)

def select_image_range(first_index, offset, num_available, num_img=None, name=None,
        num_required=None):
    if isinstance(name, str):
        name = f'{name.strip()} '
    else:
        name = ''
    # Check existing values
    if not is_int(num_available, gt=0):
        logger.warning(f'No available {name}images')
        return(0, 0, 0)
    if num_img is not None and not is_int(num_img, ge=0):
        illegal_value(num_img, 'num_img', 'select_image_range')
        return(0, 0, 0)
    if is_int(first_index, ge=0) and is_int(offset, ge=0):
        if num_required is None:
            if input_yesno(f'\nCurrent {name}first image index/offset = {first_index}/{offset},'+
                    'use these values (y/n)?', 'y'):
                if num_img is not None:
                    if input_yesno(f'Current number of {name}images = {num_img}, '+
                            'use this value (y/n)? ', 'y'):
                        return(first_index, offset, num_img)
                else:
                    if input_yesno(f'Number of available {name}images = {num_available}, '+
                            'use all (y/n)? ', 'y'):
                        return(first_index, offset, num_available)
        else:
            if input_yesno(f'\nCurrent {name}first image offset = {offset}, '+
                    f'use this values (y/n)?', 'y'):
                return(first_index, offset, num_required)

    # Check range against requirements
    if num_required is None:
        if num_available == 1:
            return(first_index, 0, 1)
    else:
        if not is_int(num_required, ge=1):
            illegal_value(num_required, 'num_required', 'select_image_range')
            return(0, 0, 0)
        if num_available < num_required:
            logger.error(f'Unable to find the required {name}images ({num_available} out of '+
                    f'{num_required})')
            return(0, 0, 0)

    # Select index range
    print(f'\nThe number of available {name}images is {num_available}')
    if num_required is None:
        last_index = first_index+num_available
        use_all = f'Use all ([{first_index}, {last_index}])'
        pick_offset = 'Pick the first image index offset and the number of images'
        pick_bounds = 'Pick the first and last image index'
        choice = input_menu([use_all, pick_offset, pick_bounds], default=pick_offset)
        if not choice:
            offset = 0
            num_img = num_available
        elif choice == 1:
            offset = input_int('Enter the first index offset', ge=0, le=last_index-first_index)
            if first_index+offset == last_index:
                num_img = 1
            else:
                num_img = input_int('Enter the number of images', ge=1, le=num_available-offset)
        else:
            offset = input_int('Enter the first index', ge=first_index, le=last_index)
            num_img = 1-offset+input_int('Enter the last index', ge=offset, le=last_index)
            offset -= first_index
    else:
        use_all = f'Use ([{first_index}, {first_index+num_required-1}])'
        pick_offset = 'Pick the first index offset'
        choice = input_menu([use_all, pick_offset], pick_offset)
        offset = 0
        if choice == 1:
            offset = input_int('Enter the first index offset', ge=0, le=num_available-num_required)
        num_img = num_required

    return(first_index, offset, num_img)

def load_image(f, img_x_bounds=None, img_y_bounds=None):
    """Load a single image from file.
    """
    if not os.path.isfile(f):
        logger.error(f'Unable to load {f}')
        return(None)
    img_read = plt.imread(f)
    if not img_x_bounds:
        img_x_bounds = (0, img_read.shape[0])
    else:
        if (not isinstance(img_x_bounds, (tuple, list)) or len(img_x_bounds) != 2 or 
                not (0 <= img_x_bounds[0] < img_x_bounds[1] <= img_read.shape[0])):
            logger.error(f'inconsistent row dimension in {f}')
            return(None)
    if not img_y_bounds:
        img_y_bounds = (0, img_read.shape[1])
    else:
        if (not isinstance(img_y_bounds, list) or len(img_y_bounds) != 2 or 
                not (0 <= img_y_bounds[0] < img_y_bounds[1] <= img_read.shape[1])):
            logger.error(f'inconsistent column dimension in {f}')
            return(None)
    return(img_read[img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]])

def load_image_stack(files, filetype, img_offset, num_img, num_img_skip=0,
        img_x_bounds=None, img_y_bounds=None):
    """Load a set of images and return them as a stack.
    """
    logger.debug(f'img_offset = {img_offset}')
    logger.debug(f'num_img = {num_img}')
    logger.debug(f'num_img_skip = {num_img_skip}')
    logger.debug(f'\nfiles:\n{files}\n')
    img_stack = np.array([])
    if filetype == 'tif':
        img_read_stack = []
        i = 1
        t0 = time()
        for f in files[img_offset:img_offset+num_img:num_img_skip+1]:
            if not i%20:
                logger.info(f'    loading {i}/{num_img}: {f}')
            else:
                logger.debug(f'    loading {i}/{num_img}: {f}')
            img_read = load_image(f, img_x_bounds, img_y_bounds)
            img_read_stack.append(img_read)
            i += num_img_skip+1
        img_stack = np.stack([img_read for img_read in img_read_stack])
        logger.info(f'... done in {time()-t0:.2f} seconds!')
        logger.debug(f'img_stack shape = {np.shape(img_stack)}')
        del img_read_stack, img_read
    elif filetype == 'h5':
        if not isinstance(files[0], str) and not os.path.isfile(files[0]):
            illegal_value(files[0], 'files[0]', 'load_image_stack')
            return(img_stack)
        t0 = time()
        logger.info(f'Loading {files[0]}')
        with h5py.File(files[0], 'r') as f:
            shape = f['entry/instrument/detector/data'].shape
            if len(shape) != 3:
                logger.error(f'inconsistent dimensions in {files[0]}')
            if not img_x_bounds:
                img_x_bounds = (0, shape[1])
            else:
                if (not isinstance(img_x_bounds, (tuple, list)) or len(img_x_bounds) != 2 or 
                        not (0 <= img_x_bounds[0] < img_x_bounds[1] <= shape[1])):
                    logger.error(f'inconsistent row dimension in {files[0]} {img_x_bounds} '+
                            f'{shape[1]}')
            if not img_y_bounds:
                img_y_bounds = (0, shape[2])
            else:
                if (not isinstance(img_y_bounds, list) or len(img_y_bounds) != 2 or 
                        not (0 <= img_y_bounds[0] < img_y_bounds[1] <= shape[2])):
                    logger.error(f'inconsistent column dimension in {files[0]}')
            img_stack = f.get('entry/instrument/detector/data')[
                    img_offset:img_offset+num_img:num_img_skip+1,
                    img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]
        logger.info(f'... done in {time()-t0:.2f} seconds!')
    else:
        illegal_value(filetype, 'filetype', 'load_image_stack')
    return(img_stack)

def combine_tiffs_in_h5(files, num_img, h5_filename):
    img_stack = load_image_stack(files, 'tif', 0, num_img)
    with h5py.File(h5_filename, 'w') as f:
        f.create_dataset('entry/instrument/detector/data', data=img_stack)
    del img_stack
    return([h5_filename])

def clear_imshow(title=None):
    plt.ioff()
    if title is None:
        title = 'quick imshow'
    elif not isinstance(title, str):
        illegal_value(title, 'title', 'clear_imshow')
        return
    plt.close(fig=title)

def clear_plot(title=None):
    plt.ioff()
    if title is None:
        title = 'quick plot'
    elif not isinstance(title, str):
        illegal_value(title, 'title', 'clear_plot')
        return
    plt.close(fig=title)

def quick_imshow(a, title=None, path=None, name=None, save_fig=False, save_only=False,
            clear=True, extent=None, show_grid=False, grid_color='w', grid_linewidth=1,
            block=False, **kwargs):
    if title is not None and not isinstance(title, str):
        illegal_value(title, 'title', 'quick_imshow')
        return
    if path is not None and not isinstance(path, str):
        illegal_value(path, 'path', 'quick_imshow')
        return
    if not isinstance(save_fig, bool):
        illegal_value(save_fig, 'save_fig', 'quick_imshow')
        return
    if not isinstance(save_only, bool):
        illegal_value(save_only, 'save_only', 'quick_imshow')
        return
    if not isinstance(clear, bool):
        illegal_value(clear, 'clear', 'quick_imshow')
        return
    if not isinstance(block, bool):
        illegal_value(block, 'block', 'quick_imshow')
        return
    if not title:
        title='quick imshow'
#    else:
#        title = re.sub(r"\s+", '_', title)
    if name is None:
        ttitle = re.sub(r"\s+", '_', title)
        if path is None:
            path = f'{ttitle}.png'
        else:
            path = f'{path}/{ttitle}.png'
    else:
        if path is None:
            path = name
        else:
            path = f'{path}/{name}'
    if 'cmap' in kwargs and a.ndim == 3 and (a.shape[2] == 3 or a.shape[2] == 4):
        use_cmap = True
        if a.shape[2] == 4 and a[:,:,-1].min() != a[:,:,-1].max():
            use_cmap = False
        if any(True if a[i,j,0] != a[i,j,1] and a[i,j,0] != a[i,j,2] else False
                for i in range(a.shape[0]) for j in range(a.shape[1])):
            use_cmap = False
        if use_cmap:
            a = a[:,:,0]
        else:
            logger.warning('Image incompatible with cmap option, ignore cmap')
            kwargs.pop('cmap')
    if extent is None:
        extent = (0, a.shape[1], a.shape[0], 0)
    if clear:
        try:
            plt.close(fig=title)
        except:
            pass
    if not save_only:
        if block:
            plt.ioff()
        else:
            plt.ion()
    plt.figure(title)
    plt.imshow(a, extent=extent, **kwargs)
    if show_grid:
        ax = plt.gca()
        ax.grid(color=grid_color, linewidth=grid_linewidth)
#    if title != 'quick imshow':
#        plt.title = title
    if save_only:
        plt.savefig(path)
        plt.close(fig=title)
    else:
        if save_fig:
            plt.savefig(path)
        if block:
            plt.show(block=block)

def quick_plot(*args, xerr=None, yerr=None, vlines=None, title=None, xlim=None, ylim=None,
        xlabel=None, ylabel=None, legend=None, path=None, name=None, show_grid=False,
        save_fig=False, save_only=False, clear=True, block=False, **kwargs):
    if title is not None and not isinstance(title, str):
        illegal_value(title, 'title', 'quick_plot')
        title = None
    if xlim is not None and not isinstance(xlim, (tuple, list)) and len(xlim) != 2:
        illegal_value(xlim, 'xlim', 'quick_plot')
        xlim = None
    if ylim is not None and not isinstance(ylim, (tuple, list)) and len(ylim) != 2:
        illegal_value(ylim, 'ylim', 'quick_plot')
        ylim = None
    if xlabel is not None and not isinstance(xlabel, str):
        illegal_value(xlabel, 'xlabel', 'quick_plot')
        xlabel = None
    if ylabel is not None and not isinstance(ylabel, str):
        illegal_value(ylabel, 'ylabel', 'quick_plot')
        ylabel = None
    if legend is not None and not isinstance(legend, (tuple, list)):
        illegal_value(legend, 'legend', 'quick_plot')
        legend = None
    if path is not None and not isinstance(path, str):
        illegal_value(path, 'path', 'quick_plot')
        return
    if not isinstance(show_grid, bool):
        illegal_value(show_grid, 'show_grid', 'quick_plot')
        return
    if not isinstance(save_fig, bool):
        illegal_value(save_fig, 'save_fig', 'quick_plot')
        return
    if not isinstance(save_only, bool):
        illegal_value(save_only, 'save_only', 'quick_plot')
        return
    if not isinstance(clear, bool):
        illegal_value(clear, 'clear', 'quick_plot')
        return
    if not isinstance(block, bool):
        illegal_value(block, 'block', 'quick_plot')
        return
    if title is None:
        title = 'quick plot'
#    else:
#        title = re.sub(r"\s+", '_', title)
    if name is None:
        ttitle = re.sub(r"\s+", '_', title)
        if path is None:
            path = f'{ttitle}.png'
        else:
            path = f'{path}/{ttitle}.png'
    else:
        if path is None:
            path = name
        else:
            path = f'{path}/{name}'
    if clear:
        try:
            plt.close(fig=title)
        except:
            pass
    args = unwrap_tuple(args)
    if depth_tuple(args) > 1 and (xerr is not None or yerr is not None):
        logger.warning('Error bars ignored form multiple curves')
    if not save_only:
        if block:
            plt.ioff()
        else:
            plt.ion()
    plt.figure(title)
    if depth_tuple(args) > 1:
       for y in args:
           plt.plot(*y, **kwargs)
    else:
        if xerr is None and yerr is None:
            plt.plot(*args, **kwargs)
        else:
            plt.errorbar(*args, xerr=xerr, yerr=yerr, **kwargs)
    if vlines is not None:
        if isinstance(vlines, (int, float)):
            vlines = [vlines]
        for v in vlines:
            plt.axvline(v, color='r', linestyle='--', **kwargs)
#    if vlines is not None:
#        for s in tuple(([x, x], list(plt.gca().get_ylim())) for x in vlines):
#            plt.plot(*s, color='red', **kwargs)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if show_grid:
        ax = plt.gca()
        ax.grid(color='k')#, linewidth=1)
    if legend is not None:
        plt.legend(legend)
    if save_only:
        plt.savefig(path)
        plt.close(fig=title)
    else:
        if save_fig:
            plt.savefig(path)
        if block:
            plt.show(block=block)

def select_array_bounds(a, x_low=None, x_upp=None, num_x_min=None, ask_bounds=False,
        title='select array bounds'):
    """Interactively select the lower and upper data bounds for a numpy array.
    """
    if isinstance(a, (tuple, list)):
        a = np.array(a)
    if not isinstance(a, np.ndarray) or a.ndim != 1:
        illegal_value(a.ndim, 'array type or dimension', 'select_array_bounds')
        return(None)
    len_a = len(a)
    if num_x_min is None:
        num_x_min = 1
    else:
        if num_x_min < 2 or num_x_min > len_a:
            logger.warning('Invalid value for num_x_min in select_array_bounds, input ignored')
            num_x_min = 1

    # Ask to use current bounds
    if ask_bounds and (x_low is not None or x_upp is not None):
        if x_low is None:
            x_low = 0
        if not is_int(x_low, ge=0, le=len_a-num_x_min):
            illegal_value(x_low, 'x_low', 'select_array_bounds')
            return(None)
        if x_upp is None:
            x_upp = len_a
        if not is_int(x_upp, ge=x_low+num_x_min, le=len_a):
            illegal_value(x_upp, 'x_upp', 'select_array_bounds')
            return(None)
        quick_plot((range(len_a), a), vlines=(x_low,x_upp), title=title)
        if not input_yesno(f'\nCurrent array bounds: [{x_low}, {x_upp}] '+
                    'use these values (y/n)?', 'y'):
            x_low = None
            x_upp = None
        else:
            clear_plot(title)
            return(x_low, x_upp)

    if x_low is None:
        x_min = 0
        x_max = len_a
        x_low_max = len_a-num_x_min
        while True:
            quick_plot(range(x_min, x_max), a[x_min:x_max], title=title)
            zoom_flag = input_yesno('Set lower data bound (y) or zoom in (n)?', 'y')
            if zoom_flag:
                x_low = input_int('    Set lower data bound', ge=0, le=x_low_max)
                break
            else:
                x_min = input_int('    Set lower zoom index', ge=0, le=x_low_max)
                x_max = input_int('    Set upper zoom index', ge=x_min+1, le=x_low_max+1)
    else:
        if not is_int(x_low, ge=0, le=len_a-num_x_min):
            illegal_value(x_low, 'x_low', 'select_array_bounds')
            return(None)
    if x_upp is None:
        x_min = x_low+num_x_min
        x_max = len_a
        x_upp_min = x_min
        while True:
            quick_plot(range(x_min, x_max), a[x_min:x_max], title=title)
            zoom_flag = input_yesno('Set upper data bound (y) or zoom in (n)?', 'y')
            if zoom_flag:
                x_upp = input_int('    Set upper data bound', ge=x_upp_min, le=len_a)
                break
            else:
                x_min = input_int('    Set upper zoom index', ge=x_upp_min, le=len_a-1)
                x_max = input_int('    Set upper zoom index', ge=x_min+1, le=len_a)
    else:
        if not is_int(x_upp, ge=x_low+num_x_min, le=len_a):
            illegal_value(x_upp, 'x_upp', 'select_array_bounds')
            return(None)
    print(f'lower bound = {x_low} (inclusive)\nupper bound = {x_upp} (exclusive)]')
    quick_plot((range(len_a), a), vlines=(x_low,x_upp), title=title)
    if not input_yesno('Accept these bounds (y/n)?', 'y'):
        x_low, x_upp = select_array_bounds(a, None, None, num_x_min, title=title)
    clear_plot(title)
    return(x_low, x_upp)

def select_image_bounds(a, axis, low=None, upp=None, num_min=None, title='select array bounds',
        raise_error=False):
    """Interactively select the lower and upper data bounds for a 2D numpy array.
    """
    a = np.asarray(a)
    if a.ndim != 2:
        illegal_value(a.ndim, 'array dimension', location='select_image_bounds',
                raise_error=raise_error)
        return(None)
    if axis < 0 or axis >= a.ndim:
        illegal_value(axis, 'axis', location='select_image_bounds', raise_error=raise_error)
        return(None)
    low_save = low
    upp_save = upp
    num_min_save = num_min
    if num_min is None:
        num_min = 1
    else:
        if num_min < 2 or num_min > a.shape[axis]:
            logger.warning('Invalid input for num_min in select_image_bounds, input ignored')
            num_min = 1
    if low is None:
        min_ = 0
        max_ = a.shape[axis]
        low_max = a.shape[axis]-num_min
        while True:
            if axis:
                quick_imshow(a[:,min_:max_], title=title, aspect='auto',
                        extent=[min_,max_,a.shape[0],0])
            else:
                quick_imshow(a[min_:max_,:], title=title, aspect='auto',
                        extent=[0,a.shape[1], max_,min_])
            zoom_flag = input_yesno('Set lower data bound (y) or zoom in (n)?', 'y')
            if zoom_flag:
                low = input_int('    Set lower data bound', ge=0, le=low_max)
                break
            else:
                min_ = input_int('    Set lower zoom index', ge=0, le=low_max)
                max_ = input_int('    Set upper zoom index', ge=min_+1, le=low_max+1)
    else:
        if not is_int(low, ge=0, le=a.shape[axis]-num_min):
            illegal_value(low, 'low', location='select_image_bounds', raise_error=raise_error)
            return(None)
    if upp is None:
        min_ = low+num_min
        max_ = a.shape[axis]
        upp_min = min_
        while True:
            if axis:
                quick_imshow(a[:,min_:max_], title=title, aspect='auto',
                        extent=[min_,max_,a.shape[0],0])
            else:
                quick_imshow(a[min_:max_,:], title=title, aspect='auto',
                        extent=[0,a.shape[1], max_,min_])
            zoom_flag = input_yesno('Set upper data bound (y) or zoom in (n)?', 'y')
            if zoom_flag:
                upp = input_int('    Set upper data bound', ge=upp_min, le=a.shape[axis])
                break
            else:
                min_ = input_int('    Set upper zoom index', ge=upp_min, le=a.shape[axis]-1)
                max_ = input_int('    Set upper zoom index', ge=min_+1, le=a.shape[axis])
    else:
        if not is_int(upp, ge=low+num_min, le=a.shape[axis]):
            illegal_value(upp, 'upp', location='select_image_bounds', raise_error=raise_error)
            return(None)
    bounds = (low, upp)
    a_tmp = np.copy(a)
    a_tmp_max = a.max()
    if axis:
        a_tmp[:,bounds[0]] = a_tmp_max
        a_tmp[:,bounds[1]-1] = a_tmp_max
    else:
        a_tmp[bounds[0],:] = a_tmp_max
        a_tmp[bounds[1]-1,:] = a_tmp_max
    print(f'lower bound = {low} (inclusive)\nupper bound = {upp} (exclusive)')
    quick_imshow(a_tmp, title=title, aspect='auto')
    del a_tmp
    if not input_yesno('Accept these bounds (y/n)?', 'y'):
        bounds = select_image_bounds(a, axis, low=low_save, upp=upp_save, num_min=num_min_save,
            title=title)
    return(bounds)

def select_one_image_bound(a, axis, bound=None, bound_name=None, title='select array bounds',
        default='y', raise_error=False):
    """Interactively select a data boundary for a 2D numpy array.
    """
    a = np.asarray(a)
    if a.ndim != 2:
        illegal_value(a.ndim, 'array dimension', location='select_one_image_bound',
                raise_error=raise_error)
        return(None)
    if axis < 0 or axis >= a.ndim:
        illegal_value(axis, 'axis', location='select_one_image_bound', raise_error=raise_error)
        return(None)
    if bound_name is None:
        bound_name = 'data bound'
    if bound is None:
        min_ = 0
        max_ = a.shape[axis]
        bound_max = a.shape[axis]-1
        while True:
            if axis:
                quick_imshow(a[:,min_:max_], title=title, aspect='auto',
                        extent=[min_,max_,a.shape[0],0])
            else:
                quick_imshow(a[min_:max_,:], title=title, aspect='auto',
                        extent=[0,a.shape[1], max_,min_])
            zoom_flag = input_yesno(f'Set {bound_name} (y) or zoom in (n)?', 'y')
            if zoom_flag:
                bound = input_int(f'    Set {bound_name}', ge=0, le=bound_max)
                clear_imshow(title)
                break
            else:
                min_ = input_int('    Set lower zoom index', ge=0, le=bound_max)
                max_ = input_int('    Set upper zoom index', ge=min_+1, le=bound_max+1)

    elif not is_int(bound, ge=0, le=a.shape[axis]-1):
        illegal_value(bound, 'bound', location='select_one_image_bound', raise_error=raise_error)
        return(None)
    else:
        print(f'Current {bound_name} = {bound}')
    a_tmp = np.copy(a)
    a_tmp_max = a.max()
    if axis:
        a_tmp[:,bound] = a_tmp_max
    else:
        a_tmp[bound,:] = a_tmp_max
    quick_imshow(a_tmp, title=title, aspect='auto')
    del a_tmp
    if not input_yesno(f'Accept this {bound_name} (y/n)?', default):
        bound = select_one_image_bound(a, axis, bound_name=bound_name, title=title)
    clear_imshow(title)
    return(bound)


class Config:
    """Base class for processing a config file or dictionary.
    """
    def __init__(self, config_file=None, config_dict=None):
        self.config = {}
        self.load_flag = False
        self.suffix = None

        # Load config file 
        if config_file is not None and config_dict is not None:
            logger.warning('Ignoring config_dict (both config_file and config_dict are specified)')
        if config_file is not None:
           self.load_file(config_file)
        elif config_dict is not None:
           self.load_dict(config_dict)

    def load_file(self, config_file):
        """Load a config file.
        """
        if self.load_flag:
            logger.warning('Overwriting any previously loaded config file')
        self.config = {}

        # Ensure config file exists
        if not os.path.isfile(config_file):
            logger.error(f'Unable to load {config_file}')
            return

        # Load config file (for now for Galaxy, allow .dat extension)
        self.suffix = os.path.splitext(config_file)[1]
        if self.suffix == '.yml' or self.suffix == '.yaml' or self.suffix == '.dat':
            with open(config_file, 'r') as f:
                self.config = safe_load(f)
        elif self.suffix == '.txt':
            with open(config_file, 'r') as f:
                lines = f.read().splitlines()
            self.config = {item[0].strip():literal_eval(item[1].strip()) for item in
                    [line.split('#')[0].split('=') for line in lines if '=' in line.split('#')[0]]}
        else:
            illegal_value(self.suffix, 'config file extension', 'Config.load_file')

        # Make sure config file was correctly loaded
        if isinstance(self.config, dict):
            self.load_flag = True
        else:
            logger.error(f'Unable to load dictionary from config file: {config_file}')
            self.config = {}

    def load_dict(self, config_dict):
        """Takes a dictionary and places it into self.config.
        """
        if self.load_flag:
            logger.warning('Overwriting the previously loaded config file')

        if isinstance(config_dict, dict):
            self.config = config_dict
            self.load_flag = True
        else:
            illegal_value(config_dict, 'dictionary config object', 'Config.load_dict')
            self.config = {}

    def save_file(self, config_file):
        """Save the config file (as a yaml file only right now).
        """
        suffix = os.path.splitext(config_file)[1]
        if suffix != '.yml' and suffix != '.yaml':
            illegal_value(suffix, 'config file extension', 'Config.save_file')

        # Check if config file exists
        if os.path.isfile(config_file):
            logger.info(f'Updating {config_file}')
        else:
            logger.info(f'Saving {config_file}')

        # Save config file
        with open(config_file, 'w') as f:
            safe_dump(self.config, f)

    def validate(self, pars_required, pars_missing=None):
        """Returns False if any required keys are missing.
        """
        if not self.load_flag:
            logger.error('Load a config file prior to calling Config.validate')

        def validate_nested_pars(config, par):
            par_levels = par.split(':')
            first_level_par = par_levels[0]
            try:
                first_level_par = int(first_level_par)
            except:
                pass
            try:
                next_level_config = config[first_level_par]
                if len(par_levels) > 1:
                    next_level_par = ':'.join(par_levels[1:])
                    return(validate_nested_pars(next_level_config, next_level_par))
                else:
                    return(True)
            except:
                return(False)

        pars_missing = [p for p in pars_required if not validate_nested_pars(self.config, p)]
        if len(pars_missing) > 0:
            logger.error(f'Missing item(s) in configuration: {", ".join(pars_missing)}')
            return(False)
        else:
            return(True)
