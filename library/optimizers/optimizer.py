import torch.optim as optim
from torch import nn

class AdamOptimizer:
    """
    封装 Adam 优化器为类
    """
    def __init__(self, model: nn.Module, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        """
        初始化 Adam 优化器
        :param model: 需要优化的模型
        :param learning_rate: 学习率，默认 0.001
        :param beta1: 一阶矩估计的指数衰减率，默认 0.9
        :param beta2: 二阶矩估计的指数衰减率，默认 0.999
        :param eps: 数值稳定性常数，默认 1e-8
        """
        self.model = model
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def get_optimizer(self):
        """
        返回 Adam 优化器实例
        :return: Adam 优化器实例
        """
        return optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2), eps=self.eps)


class SGDOptimizer:
    """
    封装 SGD 优化器为类
    """
    def __init__(self, model: nn.Module, learning_rate: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0):
        """
        初始化 SGD 优化器
        :param model: 需要优化的模型
        :param learning_rate: 学习率，默认 0.01
        :param momentum: 动量因子，默认 0.9
        :param weight_decay: 权重衰减（L2 正则化），默认 0.0
        """
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

    def get_optimizer(self):
        """
        返回 SGD 优化器实例
        :return: SGD 优化器实例
        """
        return optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)


class RMSpropOptimizer:
    """
    封装 RMSprop 优化器为类
    """
    def __init__(self, model: nn.Module, learning_rate: float = 0.001, alpha: float = 0.99, eps: float = 1e-8, weight_decay: float = 0.0, momentum: float = 0.0):
        """
        初始化 RMSprop 优化器
        :param model: 需要优化的模型
        :param learning_rate: 学习率，默认 0.001
        :param alpha: 平滑常数，默认 0.99
        :param eps: 数值稳定性常数，默认 1e-8
        :param weight_decay: 权重衰减（L2 正则化），默认 0.0
        :param momentum: 动量因子，默认 0.0
        """
        self.model = model
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum

    def get_optimizer(self):
        """
        返回 RMSprop 优化器实例
        :return: RMSprop 优化器实例
        """
        return optim.RMSprop(self.model.parameters(), lr=self.learning_rate, alpha=self.alpha, eps=self.eps, weight_decay=self.weight_decay, momentum=self.momentum)


class AdagradOptimizer:
    """
    封装 Adagrad 优化器为类
    """
    def __init__(self, model: nn.Module, learning_rate: float = 0.01, lr_decay: float = 0.0, weight_decay: float = 0.0, eps: float = 1e-10):
        """
        初始化 Adagrad 优化器
        :param model: 需要优化的模型
        :param learning_rate: 学习率，默认 0.01
        :param lr_decay: 学习率衰减率，默认 0.0
        :param weight_decay: 权重衰减（L2 正则化），默认 0.0
        :param eps: 数值稳定性常数，默认 1e-10
        """
        self.model = model
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.eps = eps

    def get_optimizer(self):
        """
        返回 Adagrad 优化器实例
        :return: Adagrad 优化器实例
        """
        return optim.Adagrad(self.model.parameters(), lr=self.learning_rate, lr_decay=self.lr_decay, weight_decay=self.weight_decay, eps=self.eps)

if __name__ == "__main__":
    # 假设我们有一个简单的模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()

    # 实例化优化器类
    adam_optimizer = AdamOptimizer(model)
    sgd_optimizer = SGDOptimizer(model)
    rmsprop_optimizer = RMSpropOptimizer(model)
    adagrad_optimizer = AdagradOptimizer(model)

    # 获取优化器实例
    adam = adam_optimizer.get_optimizer()
    sgd = sgd_optimizer.get_optimizer()
    rmsprop = rmsprop_optimizer.get_optimizer()
    adagrad = adagrad_optimizer.get_optimizer()

    # 打印优化器信息
    print("Adam Optimizer:", adam)
    print("SGD Optimizer:", sgd)
    print("RMSprop Optimizer:", rmsprop)
    print("Adagrad Optimizer:", adagrad)





