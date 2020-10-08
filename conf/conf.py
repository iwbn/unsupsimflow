from util.configbox import ConfigBox

class PhotoBaseline(ConfigBox):
    def __init__(self, *args, **kwargs):
        ConfigBox.__init__(self, *args, **kwargs)
        self['learning_rate'] = 1e-4
        self['lr_decay_rate'] = 0.5
        self['lr_decay_steps'] = 50000
        self['batch_size'] = 4
        self['train_dataset'] = "flyingchairs"
        self['val_dataset'] = "flyingchairs"
        self['weight_decay'] = 0.0
        self['pyramid_photo_weight'] = 1.0
        self['pyramid_layer_weights'] =  [0.2, 0.2, 0.2, 0.2, 0.32]
        self['max_steps'] = 200000
        self['val_steps'] = 1000
        self['backward_flow'] = True
        self['input_size'] = [300, 448]
        self['rand_crop_size'] = [300, 448]
        self['use_occ'] = False

class RGBOcc(ConfigBox):
    def __init__(self, *args, **kwargs):
        ConfigBox.__init__(self, *args, **kwargs)
        self['learning_rate'] = 1e-4
        self['lr_decay_rate'] = 0.5
        self['lr_decay_steps'] = 100000
        self['batch_size'] = 4
        self['train_dataset'] = "flyingchairs"
        self['val_dataset'] = "flyingchairs"
        self['weight_decay'] = 0.0
        self['pyramid_photo_weight'] = 1.0
        self['pyramid_layer_weights'] = [0.2, 0.2, 0.2, 0.2, 0.32]
        self['max_steps'] = 300000
        self['val_steps'] = 1000
        self['backward_flow'] = True
        self['input_size'] = [300, 448]
        self['rand_crop_size'] = [300, 448]
        self['use_occ'] = True
        self['pretrained_ckpt'] = 'ckpt/rgb_pyramid_01/ckpt-200000'


class CensusOcc(ConfigBox):
    def __init__(self, *args, **kwargs):
        ConfigBox.__init__(self, *args, **kwargs)
        self['learning_rate'] = 1e-5
        self['lr_decay_rate'] = 0.5
        self['lr_decay_steps'] = 50000
        self['batch_size'] = 4
        self['train_dataset'] = "flyingchairs"
        self['val_dataset'] = "flyingchairs"
        self['weight_decay'] = 0.0
        self['census_weight'] = 1.0
        self['max_steps'] = 200000
        self['val_steps'] = 1000
        self['backward_flow'] = True
        self['input_size'] = [300, 448]
        self['rand_crop_size'] = [300, 448]
        self['use_occ'] = True
        self['pretrained_ckpt'] = 'ckpt/rgb_occ_pyramid_01/ckpt-300000'

class CensusOccDD(ConfigBox):
    def __init__(self, *args, **kwargs):
        ConfigBox.__init__(self, *args, **kwargs)
        self['learning_rate'] = 1e-5
        self['lr_decay_rate'] = 0.5
        self['lr_decay_steps'] = 50000
        self['batch_size'] = 4
        self['train_dataset'] = "flyingchairs"
        self['val_dataset'] = "flyingchairs"
        self['weight_decay'] = 0.0
        self['census_weight'] = 1.0
        self['dd_weight'] = 1.0
        self['dd_padding'] = 48
        self['teacher_flow_path'] = 'ckpt/census_occ_01/teacher_flow_flyingchairs-200000'
        self['max_steps'] = 200000
        self['val_steps'] = 1000
        self['backward_flow'] = True
        self['input_size'] = [300, 448]
        self['rand_crop_size'] = [300, 448]
        self['use_occ'] = True
        self['pretrained_ckpt'] = 'ckpt/census_occ_01/ckpt-200000'

class Ours(ConfigBox):
    def __init__(self, *args, **kwargs):
        ConfigBox.__init__(self, *args, **kwargs)
        self['learning_rate'] = 1e-5
        self['lr_decay_rate'] = 0.5
        self['lr_decay_steps'] = 50000
        self['batch_size'] = 4
        self['train_dataset'] = "flyingchairs"
        self['val_dataset'] = "flyingchairs"
        self['weight_decay'] = 0.0
        self['reg_census_weight'] = 1.0
        self['feat_sep_weight'] = 4.0
        self['cond_smooth_weight'] = 1e-4
        self['max_steps'] = 1000
        self['val_steps'] = 1000
        self['backward_flow'] = True
        self['input_size'] = [300, 448]
        self['rand_crop_size'] = [300, 448]
        self['use_occ'] = True
        self['pretrained_ckpt'] = 'ckpt/census_occ_01/ckpt-200000'

class OursFinal(ConfigBox):
    def __init__(self, *args, **kwargs):
        ConfigBox.__init__(self, *args, **kwargs)
        self['learning_rate'] = 1e-5
        self['lr_decay_rate'] = 0.5
        self['lr_decay_steps'] = 10000
        self['batch_size'] = 4
        self['train_dataset'] = "flyingchairs"
        self['val_dataset'] = "flyingchairs"
        self['weight_decay'] = 0.0
        self['reg_census_weight'] = 1.0
        self['feat_sep_weight'] = 4.0
        self['cond_smooth_weight'] = 1e-4
        self['dd_weight'] = 1.0
        self['dd_padding'] = 48
        self['teacher_flow_path'] = 'ckpt/ours_01/teacher_flow_flyingchairs-1000'
        self['max_steps'] = 50000
        self['val_steps'] = 1000
        self['backward_flow'] = True
        self['input_size'] = [300, 448]
        self['rand_crop_size'] = [300, 448]
        self['use_occ'] = True
        self['pretrained_ckpt'] = 'ckpt/ours_01/ckpt-1000'

