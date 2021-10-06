import numpy as np
from ..common import Node

def fill_diagonal(to_be_filled, filler):
    """
    将 filler 矩阵填充在 to_be_filled 的对角线上
    """
    assert to_be_filled.shape[0] / \
        filler.shape[0] == to_be_filled.shape[1] / filler.shape[1]
    n = int(to_be_filled.shape[0] / filler.shape[0])

    r, c = filler.shape
    for i in range(n):
        to_be_filled[i * r:(i + 1) * r, i * c:(i + 1) * c] = filler

    return to_be_filled

class Add(Node):
    """
    矩阵加法：支持多个矩阵
    """
    def compute(self):
        self.value = np.mat(np.zeros(self.parents[0].shape()))
        
        for parent in self.parents:
            self.value += parent.value

    def get_jacobi(self, parent):
        # 计算当前节点对父节点的雅可比矩阵
        # 矩阵加法的雅可比矩阵是单位阵
        return np.mat(np.eye(self.dimension()))

class MatMul(Node):
    """
    矩阵乘法
    """

    def compute(self):
        assert len(self.parents) == 2 and self.parents[0].shape()[
            1] == self.parents[1].shape()[0]
        self.value = self.parents[0].value * self.parents[1].value # np.mat 的 * 被重载为了矩阵乘法

    def get_jacobi(self, parent):
        zeros = np.mat(np.zeros((self.dimension(), parent.dimension())))
        if parent is self.parents[0]:
            return fill_diagonal(zeros, self.parents[1].value.T)
        else:
            jacobi = fill_diagonal(zeros, self.parents[0].value)
            row_sort = np.arange(self.dimension()).reshape(
                self.shape()[::-1]).T.ravel()
            col_sort = np.arange(parent.dimension()).reshape(
                parent.shape()[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]

class SoftMax(Node):
    """
    SoftMax 函数
    """

    @staticmethod
    def softmax(a):
        '''
        由于该函数会重复使用，所以实现为静态方法
        '''
        c = np.max(a)
        exp_a = np.exp(a-c)
        sum_exp_a = np.sum(exp_a)
        return exp_a / sum_exp_a
    
    def compute(self):
        self.value = SoftMax.softmax(self.parents[0].value)

    def get_jacobi(self, parent):
        '''
        训练时使用 CrossEntropyWithSoftMax 节点
        '''
        raise NotImplementedError("Don't use SoftMax's get_jacobi")