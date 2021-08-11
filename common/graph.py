class Graph:
    """
    计算图类
    """

    def __init__(self):
        # 计算图内的节点列表
        self.nodes = []

    def add_node(self, node):
        """
        添加节点
        :param node:
        :return:
        """
        self.nodes.append(node);

    def clear_jacobi(self):
        """
        清除计算图中全部节点的雅可比矩阵
        :return:
        """
        for node in self.nodes:
            node.clear_jacobi()

# 全局默认计算图
default_graph = Graph()