# -*- coding: utf-8 -*-
import numpy as np

'''
General purpose functions for clean up the code in critical areas.
'''

def sort_numpy(array, col=0, order_back=False):
    """
    Sorts the columns for an entire ``ndarrray`` according to sorting
    one of them.
    
    :param array: Array to sort.
    :type array: ndarray
    :param col: Master column to sort.
    :type col: int
    :param order_back: If True, also returns the index to undo the
        new order.
    :type order_back: bool
    :returns: sorted_array or [sorted_array, order_back]
    :rtype: ndarray, list
    """
    x = array[:,col]
    sorted_index = np.argsort(x, kind = 'quicksort')
    sorted_array = array[sorted_index]
    
    if not order_back:
        return sorted_array
    else:
        n_points = sorted_index.shape[0]
        order_back = np.empty(n_points, dtype=int)
        order_back[sorted_index] = np.arange(n_points)
        return [sorted_array, order_back]

def arange_col(n, dtype=int):
    """
    Returns ``np.arange`` in a column form.
    
    :param n: Length of the array.
    :type n: int
    :param dtype: Type of the array.
    :type dtype: type
    :returns: ``np.arange`` in a column form.
    :rtype: ndarray
    """
    return np.reshape(np.arange(n, dtype = dtype), (n, 1))

def bool2index(bool_):
    """
    Returns a numpy array with the indices where bool\_ is True.
    
    :param bool_: bool array to extract Trues positions.
    :type bool_: ndarray (type=bool)
    :returns: Array with the indices where bool\_ is True.
    :rtype: ndarray    
    
    .. seealso:: :func:`index2bool`
    """
    return np.arange(bool_.shape[0])[bool_]

def index2bool(index, length=None):
    """
    Returns a numpy boolean array with Trues in the input index 
    positions.
    
    :param index: index array with the Trues positions.
    :type index: ndarray (type=int)
    :param length: Length of the returned array.
    :type length: int or None
    :returns: array with Trues in the input index positions.
    :rtype: ndarray    
    
    .. seealso:: :func:`bool2index`
    """
    if index.shape[0] == 0 and length is None:
        return np.arange(0, dtype = bool)
    if length is None: length = index.max()+1
        
    sol = np.zeros(length, dtype=bool)
    sol[index] = True
    return sol

