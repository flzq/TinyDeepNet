class Graph:
    """
    计算图类
    """

    def __init__(self):
        # 计算图内的节点列表
        self.nodes = []
        self.name_scope = None

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

    def node_count(self):
        return len(self.nodes)



def get_node_from_graph(node_name, name_scope=None, graph=None):
    if graph is None:
        graph = default_graph
    if name_scope:
        node_name = name_scope + '/' + node_name
    for node in graph.nodes:
        if node.name == node_name:
            return node
    return None



# 全局默认计算图
default_graph = Graph()