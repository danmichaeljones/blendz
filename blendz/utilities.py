from builtins import *
import numpy as np

def incrementCount(start):
    count = start
    while True:
        yield count
        count += 1
