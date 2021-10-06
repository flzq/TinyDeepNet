from TinyDeepNet.ops.ops import SoftMax
import numpy as np

from ..common import Node
from ..ops import SoftMax


class CrossEntropyWithSoftMax(Node):
    """
    计算带有SoftMax的交叉熵
    """
    def compute(self):
        self.prob = SoftMax.softmax(self.parents[0].value)
        delta = 1e-7 # 防止溢出
        self.value = np.mat(-np.sum(np.multiply(self.parents[1].value, np.log(self.prob+delta))))

    def get_jacobi(self, parent):
        if parent is self.parents[0]:
            return (self.prob - self.parents[1].value).T
        else:
            return (-np.log(self.prob)).T