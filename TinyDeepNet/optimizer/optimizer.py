import abc
import numpy as np

from ..common import Node, Variable
from ..common.graph import Graph, get_node_from_graph


class Optimizer(object):
    """
    优化器基类：基于不同的梯度更新策略更新变量节点的 value
    """

    def __init__(self, graph, target, learning_rate=0.01):
        """
        graph: 计算图对象，
        target: 目标节点对象
        learning_rate: 学习率
        """
        assert isinstance(target, Node) and isinstance(graph, Graph)
        self.graph = graph
        self.target = target
        self.learning_rate = learning_rate

        # 为每个参与训练的节点累加一个Mini Batch的全部样本的梯度
        self.acc_gradient = dict()
        self.acc_no = 0

    def one_step(self):
        """
        计算并累加样本的梯度
        优化器类的执行入口
        """
        self.forward_backward()
        self.acc_no += 1

    def get_gradient(self, node):
        """
        返回样本的平均梯度
        """
        assert node in self.acc_gradient
        return self.acc_gradient[node] / self.acc_no

    @abc.abstractmethod
    def _update(self):
        """
        抽象方法，执行具体的梯度更新算法，由子类实现
        """

    def apply_gradients(self, node_gradients_dict, summarize=False, acc_no=None):

        for node, gradient in node_gradients_dict.items():
            if isinstance(node, Node):
                pass
            else:
                target_node = get_node_from_graph(node)
                assert target_node is not None
                assert self.acc_gradient[target_node].shape == gradient.shape
                if summarize:
                    self.acc_gradient[target_node] += gradient
                else:
                    self.acc_gradient[target_node] = gradient

        if summarize:
            self.acc_no += acc_no
        else:
            if acc_no is None:
                # 传入的是平均梯度, 强制让acc_no变为1，避免梯度更新时重复平均
                self.acc_no = 1
            else:
                self.acc_no = acc_no

    def update(self, var_gradients=None):

        if var_gradients is not None:
            self.apply_gradients(var_gradients)

        # 执行更新
        self._update()

        # 清除累加梯度
        self.acc_gradient.clear()
        self.acc_no = 0

    def forward_backward(self):
        """
        首先完成一次前向传播，计算结果节点的值；
        之后进行反向传播，计算结果节点对各个节点的雅可比矩阵；
        """

        # 清除计算图中所有节点的雅可比矩阵
        self.graph.clear_jacobi()

        # 前向传播计算结果节点
        self.target.forward()

        # 反向传播计算雅可比矩阵
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                node.backward(self.target)

                # 最终结果（标量）对节点值的雅可比是一个行向量，其转置是梯度（列向量）
                # 这里将梯度reshape成与节点值相同的形状，好对节点值进行更新。
                gradient = node.jacobi.T.reshape(node.shape())
                if node not in self.acc_gradient:
                    self.acc_gradient[node] = gradient
                else:
                    self.acc_gradient[node] += gradient


class GradientDescent(Optimizer):
    """
    梯度下降优化器：朴素梯度下降法--批量梯度下降法
    """

    def __init__(self, graph, target, learning_rate=0.01):

        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate

    def _update(self):
        """
        朴素梯度下降法
        优化器重写_update方法
        """
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:

                # 取得该节点在当前小批量数据集中的平均梯度
                gradient = self.get_gradient(node)

                # 用朴素梯度下降法更新变量节点的值
                node.set_value(node.value - self.learning_rate * gradient)



class Momentum(Optimizer):
    """
    冲量法：通过累积历史梯度，调整当前的梯度方向
    """

    def __init__(self, graph, target, learning_rate=0.01, momentum=0.9):

        Optimizer.__init__(self, graph, target)

        self.learning_rate = learning_rate

        # 衰减系数，默认为0.9
        self.momentum = momentum

        # 积累历史速度的字典
        self.v = dict()

    def _update(self):

        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:

                # 取得该节点在当前batch_size的平均梯度
                gradient = self.get_gradient(node)

                if node not in self.v:
                    self.v[node] = gradient
                else:
                    # 通过历史梯度调整当前梯度
                    self.v[node] = self.momentum * self.v[node] \
                        - self.learning_rate * gradient

                # 更新变量节点的值
                node.set_value(node.value + self.v[node])


class AdaGrad(Optimizer):
    """
    AdaGrad优化器对梯度的每个分量采用了不同的学习率
    """
    def __init__(self, graph, target, learning_rate=0.01):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate
        # 累加历史梯度各分量的平方的字典
        self.s = dict()
    
    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                # 获得当前节点的平均梯度
                gradient = self.get_gradient(node)

                # 累加历史梯度的平方和
                if node not in self.s:
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.s[node] = self.s[node] + np.power(gradient, 2)
                
                # 更新当前节点参数
                node.set_value(node.value - self.learning_rate * gradient / (np.sqrt(self.s[node]+1e-10)))