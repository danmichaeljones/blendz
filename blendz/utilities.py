import numpy as np

def incrementCount(start):
    count = start
    while True:
        yield count
        count += 1


class Reject(object):
    def __init__(self, function, x_min, x_max, y_max=None, y_test_len=1000, seed=None):
        self.function = function
        self.x_min = x_min
        self.x_max = x_max
        self.rstate = np.random.RandomState(seed)

        if y_max is None:
            x_grid = np.linspace(x_min, x_max, y_test_len)
            try:
                y_grid = self.function(x_grid)
            except:
                y_grid = np.array([self.function(x) for x in x_grid])
            self.y_max = np.max(y_grid)
        else:
            self.y_max = y_max

    def sample(self, N):
        out = np.zeros(N)
        for n in xrange(N):
            accept = False
            while not accept:
                rand_x = self.rstate.uniform(self.x_min, self.x_max)
                rand_y = self.rstate.uniform(0, self.y_max)
                if self.function(rand_x) >= rand_y:
                    out[n] = rand_x
                    accept = True
        return out
