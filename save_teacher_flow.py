import tensorflow as tf
import argparse
import os
from model.feature_model import FeatureModel
from data.flyingchairs import FlyingChairsDD
from data.sintel import SintelDD
import data.data_util as data_util
from util.configbox import ConfigBox
from util.flow import visualize_flow
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Save teacher flow for datasets')
parser.add_argument('--ckpt_path', metavar='ckpt_path', type=str,
                    help='model ckpt path', required=True)
parser.add_argument('--dataset', metavar='dataset', type=str,
                    help='dataset name', required=True)

args = parser.parse_args()
print(args.ckpt_path)
print(args.dataset)

model_dir = "/".join(args.ckpt_path.split('/')[:-1])
steps = args.ckpt_path.split('/')[-1].split("-")[-1]
teacher_flow_dir = os.path.join(model_dir, "teacher_flow_%s-%s" % (args.dataset, steps))

print(teacher_flow_dir)
with open(os.path.join(model_dir, "params.yaml")) as f:
    params = ConfigBox.from_yaml(f.read())

params.teacher_flow_path = teacher_flow_dir
model = FeatureModel(params)
if args.dataset.lower() == "flyingchairs":
    data = FlyingChairsDD(params)
    data.initialize()
    im = data.get("training_data")
    flow = data.get("training_flow")
    data_fname_fw = data.get("training_data_teacher_flow_fname")
    data_fname_bw = data.get("training_data_teacher_flow_fname_bw")
    dataset = tf.data.Dataset.zip((im, flow, data_fname_fw, data_fname_bw))
    keys = [["im"], ["flow", "fname_fw", "fname_bw"]]

elif args.dataset.lower() == "sintel":
    data = SintelDD(params)
    data.initialize()

    clean_im = data.get("training_clean")
    clean_flow = data.get("training_flow")
    clean_fname_fw = data.get("training_clean_teacher_flow_fname")
    clean_fname_bw = data.get("training_clean_teacher_flow_bw_fname")
    final_im = data.get("training_final")
    final_flow = data.get("training_flow")
    final_fname_fw = data.get("training_final_teacher_flow_fname")
    final_fname_bw = data.get("training_final_teacher_flow_bw_fname")

    dataset_clean = tf.data.Dataset.zip((clean_im, clean_flow, clean_fname_fw, clean_fname_bw))
    dataset_final = tf.data.Dataset.zip((final_im, final_flow, final_fname_fw, final_fname_bw))

    flow = data.get("training_flow")
    dataset = dataset_clean.concatenate(dataset_final)
    keys = [["im"], ["flow", "fname_fw", "fname_bw"]]

else:
    raise NotImplementedError

dataset = data_util.convert_feat_label_to_dict(dataset, keys)
dataset = dataset.batch(1)


@tf.function
def estimate_flow(inputs):
    outputs = model.feed_step(inputs)
    flow_fw = outputs['flow_fw_full'][0]
    flow_bw = outputs['flow_bw_full'][0]
    return flow_fw, flow_bw


TAG_STRING = 'PIEH'
def save_flow(flow, filename):
    assert type(filename) is str, "file is not str %r" % str(filename)
    assert filename[-4:] == '.flo', "file ending is not .flo %r" % filename[-4:]

    height, width, nBands = flow.shape
    assert nBands == 2, "Number of bands = %r != 2" % nBands
    u = flow[: , : , 0]
    v = flow[: , : , 1]
    assert u.shape == v.shape, "Invalid flow shape"
    height, width = u.shape

    f = open(filename,'wb')
    f.write(TAG_STRING.encode())
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    tmp = np.zeros((height, width, nBands), dtype=np.float32)
    tmp[:,:, 0] = u
    tmp[:,:, 1] = v
    tmp.astype(np.float32).tofile(f)

    f.close()


for d in dataset:
    x, y = d
    flow_fw, flow_bw = estimate_flow(d)
    cv2.imshow("fw", visualize_flow(flow_fw).numpy())
    cv2.imshow("bw", visualize_flow(flow_bw).numpy())

    fname_fw = y['fname_fw'][0,0]
    fname_bw = y['fname_bw'][0,0]
    cv2.waitKey(1)
    os.makedirs(os.path.split(fname_fw.numpy())[0], exist_ok=True)
    os.makedirs(os.path.split(fname_bw.numpy())[0], exist_ok=True)

    save_flow(flow_fw.numpy(), fname_fw.numpy().decode('utf-8'))
    save_flow(flow_bw.numpy(), fname_bw.numpy().decode('utf-8'))

    print(fname_fw.numpy().decode('utf-8'), fname_bw.numpy().decode('utf-8'))
