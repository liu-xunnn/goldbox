from functools import reduce
import random
from numpy import exp


def sigmoid(output):
    return 1 / (1 + exp(-output))


# 创建节点对象
class Node:
    def __init__(self, layer_index, node_index):
        """
        :param layer_index: 层号
        :param node_index: 序号
        """
        self.layer_index = layer_index
        self.node_index = node_index
        self.up_stream = []
        self.down_stream = []
        self.output = 0
        self.delta = 0

    def append_upstream_connection(self, connection):
        """
        为节点添加一个上游连接
        :param connection: 要加入的连接
        :return:
        """
        self.up_stream.append(connection)

    def append_downstream_connection(self, connection):
        """
        为节点添加一个下游连接
        :param connection: 要加入的连接
        :return:
        """
        self.down_stream.append(connection)

    def set_output(self, output):
        """
        当节点在输入层，设置节点输出值
        :param output:
        :return:
        """
        self.output = output

    def calculate_output(self):
        """
        计算节点输出值
        :return:
        """
        output = reduce(lambda x, connection: x + connection.up_stream_node.output * connection.weight,
                        self.up_stream, 0)
        self.output = sigmoid(output)

    def calculate_outlayer_delta(self, label):
        """
        计算输出层误差值
        :return:
        """
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def calculate_hiddenlayer_delta(self):
        """
        计算隐藏层误差值
        :return:
        """
        downstream_sum = reduce(lambda x, connection: x + connection.down_stream_node.delta * connection.weight,
                                self.down_stream, 0)
        self.delta = self.output * (1 - self.output) * downstream_sum

    def __str__(self):
        """
        打印节点信息：层号，序号，输出，梯度，上游连接，下游连接
        :return:
        """
        node_str = '%d-%d output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        return node_str


# 输出恒为1的节点,计算偏置项需要
class ConstNode:
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.output = 1
        self.down_stream = []

    def append_downstream(self, connection):
        """
        为节点添加一个下游的连接
        :param connection:
        :return:
        """
        self.down_stream.append(connection)

    def __str__(self):
        """
        打印节点信息
        :return:
        """
        node_str = '%d-%d output: %f' % (self.layer_index,
                                         self.node_index,
                                         self.output)
        return node_str


# 创建层对象
class Layer:
    def __init__(self, layer_index, node_count):
        """
        层的初始化
        :param layer_index:层号
        :param node_count:序号
        """
        self.layer_index = layer_index
        self.node_count = node_count
        self.layer_nodes = []
        for i in range(node_count-1):
            self.layer_nodes.append(Node(layer_index, i))
        # 每层最后一个是输出恒为1的节点
        self.layer_nodes.append(ConstNode(layer_index, node_count - 1))

    def set_inlayer_output(self, data):
        """
        设置输入层节点的输出
        :param data:
        :return:
        """
        for node in self.layer_nodes[:-1]:
            for i in range(len(data)):
                node.output = data[i]

    def get_output(self):
        """
        得到输出层的输出
        :return:
        """
        return [node.output for node in self.layer_nodes[:-1]]

    def __str__(self):
        layer_str = '层号：%d 节点数：%d' % (self.layer_index, self.node_count)
        nodes_str = reduce(lambda x, node: x + '\n\t' + str(node), self.layer_nodes, '')
        return '-' * 20 + '\n\t' + layer_str + '\n\t所有节点信息：' + nodes_str


# 创建连接对象
class Connection:
    def __init__(self, up_stream_node, down_stream_node):
        self.weight = random.uniform(-0.1, 0.1)
        self.up_stream_node = up_stream_node
        self.down_stream_node = down_stream_node
        self.gradient = 0

    def cal_gradient(self):
        """
        计算对应连接的梯度
        :return:
        """
        self.gradient = self.down_stream_node.delta * self.up_stream_node.output

    def update_weight(self, rate):
        """
        更新权值
        :param rate:
        :return:
        """
        self.weight = self.weight + rate * self.gradient

    def __str__(self):
        """
        打印连接信息
        :return:
        """
        return '%d-%d -> %d-%d(weight: %f  gradient: %f)' % (self.up_stream_node.layer_index,
                                                             self.up_stream_node.node_index,
                                                             self.down_stream_node.layer_index,
                                                             self.down_stream_node.node_index,
                                                             self.weight,
                                                             self.gradient)


# 创建连接集对象
class Connections:
    def __init__(self):
        self.connections = []

    def append_connection(self, connection):
        """
        增加一个连接
        :param connection:
        :return:
        """
        self.connections.append(connection)

    def __str__(self):
        return reduce(lambda x, connection: x + '\n\t' + str(connection),
                      self.connections, '')


# 创建神经网络对象
class Network:
    def __init__(self, node_count):
        """
        初始化神经网络
        :param node_count: 每层节点数
        """
        self.layers = []
        self.layers_number = len(node_count)
        self.connections = Connections()
        # 初始化层
        for i in range(self.layers_number):
            self.layers.append(Layer(i, node_count[i]))
        # 初始化连接
        for i in range(self.layers_number-1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[i].layer_nodes
                           for downstream_node in self.layers[i+1].layer_nodes[:-1]]
            for connection in connections:
                self.connections.append_connection(connection)
                connection.down_stream_node.up_stream.append(connection)
                connection.up_stream_node.down_stream.append(connection)

    def update_output(self, data_vec):
        """
        输入一个x向量并更新神经网络所有节点的输出
        :param data_vec:
        :return:
        """
        self.layers[0].set_inlayer_output(data_vec)
        for layer in self.layers[1:]:
            for node in layer.layer_nodes[:-1]:
                node.calculate_output()

    def update_delta(self, label_vec):
        """
        更新神经网络所有节点的误差值
        :return:
        """
        # 更新输出层节点误差值
        for i in range(len(label_vec)):
            self.layers[-1].layer_nodes[i].calculate_outlayer_delta(label_vec[i])
        # 更新隐藏层节点误差值
        for layer in self.layers[-2:0:-1]:
            for node in layer.layer_nodes[:-1]:
                node.calculate_hiddenlayer_delta()

    def update_weight(self, rate):
        for connection in self.connections.connections[-1::-1]:
            connection.cal_gradient()
            connection.update_weight(rate)

    def train_one_sample(self, data, label, rate):
        self.update_output(data)
        self.update_delta(label)
        self.update_weight(rate)

    def train(self, datas, labels, iteration, rate):
        for i in range(iteration):
            for j in range(len(datas)):
                self.train_one_sample(datas[j], labels[j], rate)

    def predict(self, data_vec):
        """
        预测
        :param data_vec:
        :return:
        """
        self.update_output(data_vec)
        result = self.layers[-1].get_output()
        return result

    def __str__(self):
        """
        打印神经网络信息
        :return:
        """
        layer_str = reduce(lambda x, layer: x + '\n' + str(layer), self.layers, '')
        connections_str = str(self.connections)
        return layer_str + '\n' + connections_str


if __name__ == '__main__':
    network = Network([3, 3, 3])
    print(network)
    network.train([(1, 1)], [(0.5, 0.5)], 1000, 0.1)
    result = network.predict((1, 1))
    print(result)