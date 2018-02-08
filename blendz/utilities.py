from builtins import *
import os
import sys
import numpy as np

from contextlib import contextmanager

def incrementCount(start):
    count = start
    while True:
        yield count
        count += 1

class _silentWriter(object):
    def write(self, message):
        pass

class Silence(object):
    '''
    Context manager to silence any output except
    when using the returned override function.

    Usage:

    >> with Silence() as override:
    >>     print(1)
    >>     with override():
    >>         print(2)
    >>     print(3)

    2
    '''
    #https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python
    #https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto

    @contextmanager
    def _override(self):
        sys.stdout = self._original_stdout
        yield
        sys.stdout = _silentWriter()

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = _silentWriter()
        return self._override

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
