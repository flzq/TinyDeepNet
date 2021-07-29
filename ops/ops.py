import numpy as np
from ..common import Node

class Add(Node):
    def compute(self):
        self.value = np.mat(np.zeros(self.parents[0].shape))
        
        for parent in self.parents:
            self.value += parent.value