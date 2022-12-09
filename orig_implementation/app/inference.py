# inference.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)
#

from abc import ABC, abstractmethod
from typing import Tuple, Sequence

class Inference(ABC):
    def __init__(self, rows, cols):
        self._rows = rows
        self._cols = cols
    @property
    def n_rows(self) -> int:
        """ Get the number of rows of the discretized map.
        """
        return self._rows
    @property
    def n_cols(self) -> int:
        """ Get the number of cols of the discretized map.
        """
        return self._cols
    def get_coordinate(self, r: int, c: int) -> (float, float):
        """ Get the center coordinate (x, y) of a given cell.
        """
        return (c+0.5)/self.n_cols, (self.n_rows-1-r+0.5)/self.n_rows
    def transition_model(self, r: int, c: int) -> Sequence[Tuple[float, Tuple[int, int]]]:  
        """ Get the transition probability from (r, c) to a new cell (r_, c_).

        Returns
        -------
        An iterable list of (probability, r_, c_).
        """
        r_, c_ = [], []
        if r > 0: r_.append(r-1)
        r_.append(r)
        if r < self.n_rows-1: r_.append(r+1)   

        if c > 0: c_.append(c-1)
        c_.append(c)
        if c < self.n_cols-1: c_.append(c+1)   
        
        p = 1. / (len(r_)*len(c_))
        return [(p, (r, c)) for r in r_ for c in c_]

    def normalize(self, belief: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]:
        """ Normalize a 2d belief array such that the sum of all its elements is equal to 1. 
        Returns
        -------
        A normalized belief array.
        """
        s = sum(sum(r) for r in belief)
        if s:
            return [[c/s for c in r] for r in belief]
        p = 1./sum(len(_) for _ in range(len(belief)))
        return [[p]*len(_) for _ in range(len(belief))]

    @abstractmethod
    def observe(self,
            observed_distances: Sequence[float],
            landmarks: Sequence[Tuple[float, float]]
        ):
        raise NotImplementedError
    @abstractmethod
    def timeUpdate(self):
        raise NotImplementedError