import abc


class BaseNet(metaclass=abc.ABCMeta):
    def __init__(self, input_layer, is_train):
        self.input_layer = input_layer
        self.is_train = is_train
        self.arch_output = None
        self.embedding = None
