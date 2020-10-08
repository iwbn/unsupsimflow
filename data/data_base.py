import abc
import tensorflow as tf

class Data:
    __metaclass__ = abc.ABCMeta

    # NOT YET IMPLEMENTED. YOU MIGHT LATER IMPLEMENT THIS IF NEEDED.
    def __init__(self, name):
        self.name = name
        self._datasets = {}  # dictionary with tf.data.Dataset elements
        self._initialized = False

    def initialize(self):
        if not self.initialized:
            self.prepare()
        else:
            raise
        self._initialized = True

    @abc.abstractmethod
    def prepare(self, *args):
        pass

    def get(self, key):
        return self._datasets[key]

    def set(self, key, dataset):
        self._datasets[key] = dataset

    @property
    def keys(self):
        return list(self._datasets.keys())

    @property
    def initialized(self):
        return self._initialized

