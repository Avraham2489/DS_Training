import pandas as pd
import time
import random

slow_factor = 0.1

def do_slow_stuff(output_func):
    l = 5.0 / slow_factor
    for i in xrange(int(l)):
        output_func(i / l)
        time.sleep(slow_factor)
    output_func(1.0)

def slow_calc():
    start_t = time.clock()
    end_t = start_t + (random.random() / slow_factor)
    while time.clock() < end_t:
        continue

class SlowWrapper():
    def __init__(self, func):
        self._func = func
        
    def __call__(self, *a, **b):
        time.sleep(slow_factor)
        return (self._func(*a, **b))

class SlowSeries(pd.core.series.Series):
    def apply(self, func, convert_dtype=True, args=(), **kwds):
        return super(SlowSeries, self).apply(SlowWrapper(func), convert_dtype=True, args=(), **kwds)

class SlowDataFrame(pd.DataFrame):
    def __getitem__(self, key):
        result = super(SlowDataFrame, self).__getitem__(key)
        result.__class__ = SlowSeries
        return result
    
    def apply(self, func, axis=0, broadcast=False, raw=False, reduce=None, args=(), **kwds):
        return super(SlowDataFrame, self).apply(SlowWrapper(func), axis=axis, broadcast=broadcast, raw=False, reduce=None, args=(), **kwds)
