import argparse
import importlib
import conf.conf as conf
from model.feature_model import FeatureModel

parser = argparse.ArgumentParser(description='Train feature model')
parser.add_argument('--ckpt_path', metavar='ckpt_path', type=str,
                    help='model ckpt path', required=True)
parser.add_argument('--conf_cls', metavar='conf_cls', type=str,
                    help='configuration class name (see in conf/conf.py)', required=True)
args = parser.parse_args()

Params = getattr(conf, args.conf_cls)

params = Params()
params.ckpt_path = args.ckpt_path
params.save_or_load()


model = FeatureModel(params)

model.train()