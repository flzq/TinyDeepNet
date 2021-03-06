"""
计算图中的节点类
"""
import abc
import numpy as np
from .graph import Graph, default_graph

class Node(object):
    def __init__(self, *parents, **kwargs):
        # 计算图对象，默认为全局对象 default_graph
        self.graph = kwargs.get('graph', default_graph)
        self.need_save = kwargs.get('need_save', True)
        self.gen_node_name(**kwargs)

        self.parents = list(parents)  # 父节点列表
        self.children = []  # 子节点列表
        self.value = None  # 本节点值
        self.jacobi = None  # 结果节点对本节点的雅可比矩阵
        
        # 将本节点添加到父节点的子节点列表中
        for parent in parents:
            parent.children.append(self)

        # 将本节点添加到计算图中
        self.graph.add_node(self)
        
    def get_parents(self):
        """
        :return: 返回本节点的父节点
        """
        return self.parents
    
    def get_children(self):
        """
        :return: 返回本节点的子节点
        """
        return self.children

    def gen_node_name(self, **kwargs):
        """
        生成节点名称，如果用户不指定，则根据节点类型生成类似于"MatMul:3"的节点名，
        如果指定了name_scope，则生成类似"Hidden/MatMul:3"的节点名
        """
        self.name = kwargs.get('name', '{}:{}'.format(
            self.__class__.__name__, self.graph.node_count()))
        if self.graph.name_scope:
            self.name = '{}/{}'.format(self.graph.name_scope, self.name)
    
    def forward(self):
        '''
        前向传播计算本节点的值
        递归实现前向传播，从而计算本节点值
        :return:
        '''
        for node in self.parents:
            if node.value is None:
                node.forward()
        
        self.compute()
        
    def backward(self, result):
        '''
        反向传播，计算结果节点对本节点的雅可比矩阵
        :param result: 结果节点
        :return:
        '''
        if self.jacobi is None:
            if self is result: # 结果节点对自己的雅可比矩阵为一个单位阵
                self.jacobi = np.mat(np.eye(self.dimension()))
            else:
                # 结果节点对本节点的雅可比矩阵维度为：[1, n]
                self.jacobi = np.mat(
                    np.zeros((result.dimension(), self.dimension()))
                )
                # 当有多个子节点时， 最终结果节点对当前节点的雅可比矩阵为：结果节点对各个子节点的雅可比矩阵与子节点对当前节点的雅可比矩阵相乘，将所有乘积相加
                for child in self.get_children():
                    # 子节点的value不为None，说明它在本次的计算路径上
                    if child.value is not None:
                        # child.backward(result)：结果节点对子节点的雅可比矩阵
                        # child.get_jacobi(self)：子节点对当前节点的雅可比矩阵
                        self.jacobi += child.backward(result) * child.get_jacobi(self)
        
        return self.jacobi
        
        
    
    @abc.abstractmethod
    def compute(self):
        """
        根据父节点的值计算本节点的值
        :return:
        """
    @abc.abstractmethod
    def get_jacobi(self, parent):
        """
        计算本节点对某个父节点的雅可比矩阵
        :param parent:
        :return:
        """
    
    def clear_jacobi(self):
        """
        清空结果节点对本节点的雅可比矩阵
        :return:
        """
        self.jacobi = None
        
    def dimension(self):
        """

        :return: 返回本节点的值按行展开后的向量的维度
        """
        return self.value.shape[0] * self.value.shape[1]

    def shape(self):
        """
    
        :return: 返回本节点的值作为矩阵的形状
        """
        return self.value.shape
    
    def reset_value(self, recursive=True):
        """
        重置本节点的值，同时对于子节点，因为也依赖本节点的value，本节点的值为None，
        则所有依赖本节点的下游节点的值都失效了，所以递归调用下游节点的reset_value方法，
        将所有节点的值都置为None
        :param recursive:
        :return:
        """
        self.value = None
        if recursive:
            for child in self.children:
                child.reset_value()
                

class Variable(Node):
    def __init__(self, dim, init=False, trainable=True, **kwargs):
        """
        变量节点没有父节点
        :param dim: 变量的形状
        :param init: 是否初始化
        :param trainable: 是否参与训练
        :param kwargs:
        """
        super(Variable, self).__init__()
        self.dim = dim
        
        # 如果需要初始化，通过正态分布随机初始化变量的值
        if init:
            self.value = np.mat(np.random.normal(0, 0.001, self.dim))
            
        # 变量节点是否参与训练
        self.trainable = trainable
        
    def set_value(self, value):
        """
        为变量赋值
        :param value:
        :return:
        """
        assert isinstance(value, np.matrix) and value.shape == self.dim
        
        # 本节点的值被改变，终止所有子孙节点的值
        self.reset_value()
        self.value = value