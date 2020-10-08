import abc
import tensorflow as tf
from box import Box
import os

class Model:
    __metaclass__ = abc.ABCMeta

    def __init__(self, params):
        self.params = params
        self.global_step = tf.Variable(0, tf.int32)

        self.sub_models = dict()
        self.obj_to_save = dict()
        self.model = self.model_def()
        self.initialize()

        if len(self.obj_to_save.keys()) == 0:
            print("please define obj_to_save")
            raise

        self.obj_to_save['global_step'] = self.global_step

        self.ckpt = tf.train.Checkpoint(**self.obj_to_save)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, directory=self.params['ckpt_path'], max_to_keep=50
        )

        latest_checkpoint = tf.train.latest_checkpoint(self.params['ckpt_path'])
        if latest_checkpoint is not None:
            status = self.ckpt.restore(latest_checkpoint)
        else:
            print("No checkpoint found")

    @abc.abstractmethod
    def model_def(self):
        return tf.keras.Model()

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def saver_config(self):
        pass

    @abc.abstractmethod
    def train_and_validate(self):
        pass

    def validate(self):
        pass

    def evaluate(self, dataset):
        pass

    def save(self):
        self.ckpt_manager.save(self.global_step)

    def restore(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = self.ckpt_manager.latest_checkpoint
        self.ckpt.restore(ckpt_path)



