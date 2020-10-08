import tensorflow as tf
import glob, os
import util.ops
import util.image
from data.data_base import Data
import conf.path_config as pf
import util.flow
import data.data_util
import numpy as np
import cv2

BASE_PATH = pf.get_path('FlyingChairsBasePath')
TRAIN_VAL_TXT = pf.get_path('FlyingChairsMetaFilePath')
#TRAIN_VAL_TXT =
NUM_DATA = 22872
IMG_SIZE = (384, 512)

def read_ppm(filename):
    filename = filename.decode("utf-8")
    img = cv2.imread(filename).astype('float32') / 255.
    img = img[:,:,[2,1,0]]
    return img


class FlyingChairs(Data):
    def __init__(self, params):
        Data.__init__(self, "FlyingChairs")
        self.params = params

    def prepare(self):
        all_scenes = ["%05d"%i for i in range(1, NUM_DATA+1)]

        fname_list_dict = {}
        for scene in all_scenes:
            fname_list_dict[scene] = [os.path.join(BASE_PATH,
                                                   "%s_img%d.ppm"%(scene, i)) for i in [1,2]]

        tr_scenes = set()
        ts_scenes = set()
        with open(TRAIN_VAL_TXT, 'r') as f:
            for i, l in enumerate(f):
                if int(l) == 1:
                    tr_scenes.add("%05d" % (i + 1))
                else:
                    ts_scenes.add("%05d" % (i + 1))

        data_list = []
        tr_idx = []
        ts_idx = []
        idx = 0

        for scene in all_scenes:
            for fnum, fname in enumerate(fname_list_dict[scene]):
                data_list.append(fname)
                if fnum <= len(fname_list_dict[scene]) - 2:
                    if scene in tr_scenes:
                        tr_idx.append(idx)
                    else:
                        ts_idx.append(idx)
                idx += 1

        tr_indices = tf.data.Dataset.from_tensor_slices(tr_idx)
        ts_indices = tf.data.Dataset.from_tensor_slices(ts_idx)

        def _get_img_names(indices_dataset):
            get_fnames = lambda idx: np.array([data_list[idx+i] for i in range(2)])
            dataset = indices_dataset.map(lambda idx:
                                          tf.numpy_function(get_fnames, [idx], tf.string),
                                          num_parallel_calls=32)
            return dataset

        def _get_flow_names(indices_dataset):
            get_fnames = lambda idx: np.array([_get_flow_of_frame(data_list[idx + i]) for i in range(1)])
            dataset = indices_dataset.map(lambda idx:
                                          tf.numpy_function(get_fnames, [idx], tf.string),
                                          num_parallel_calls=32)
            return dataset

        def _get_imgs(fname_dataset):
            dataset = fname_dataset.map(lambda fname: util.ops.per_batch_op(fname, _decode_ppm))
            return dataset

        def _get_flows(fname_dataset):
            dataset = fname_dataset.map(lambda fname: util.ops.per_batch_op(fname, _decode_flo))
            return dataset

        def get_datasets(indices_dataset, return_names=False):
            img_fnames = _get_img_names(indices_dataset)
            flo_fnames = _get_flow_names(indices_dataset)

            imgs = _get_imgs(img_fnames)
            flos = _get_flows(flo_fnames)
            if return_names:
                return imgs, flos, img_fnames, flo_fnames
            else:
                return imgs, flos

        tdata, tflow, tdata_fname, tflow_fname = get_datasets(tr_indices, True)
        tsdata, tsflow, tsdata_fname, ts_flow_fname = get_datasets(ts_indices, True)


        self.set("training_data", tdata)
        self.set("training_flow", tflow)

        self.set("training_data_fname", tdata_fname)
        self.set("training_flow_fname", tflow_fname)

        self.set("testing_data", tsdata)
        self.set("testing_flow", tsflow)

        self.set("testing_data_fname", tsdata_fname)
        self.set("testing_flow_fname", tsdata_fname)

    def get_relative_path_of(self, abspath):
        return os.path.relpath(abspath, BASE_PATH)


class FlyingChairsDD(FlyingChairs):
    def prepare(self):
        FlyingChairs.prepare(self)
        fname_train = self.get("training_data_fname")

        ## load teacher flow
        teacher_flow_base_path = self.params.teacher_flow_path

        _get_new = lambda x: tf.numpy_function(
            lambda orig_path: self.get_new_flo_path(BASE_PATH, orig_path, teacher_flow_base_path),
            [x], tf.string
        )

        _get_new_bw = lambda x: tf.numpy_function(
            lambda orig_path: self.get_new_flo_path(BASE_PATH, orig_path, teacher_flow_base_path, "_bw"),
            [x], tf.string
        )

        fname_t_fw = fname_train.map(lambda fnames: util.ops.per_batch_op(fnames[:-1], _get_new, tf.string))
        fname_t_bw = fname_train.map(lambda fnames: util.ops.per_batch_op(fnames[:-1], _get_new_bw, tf.string))

        self.set("training_data_teacher_flow_fname", fname_t_fw)
        self.set("training_data_teacher_flow_fname_bw", fname_t_bw)

        teacher_fw = fname_t_fw.map(lambda fname: util.ops.per_batch_op(fname, _decode_flo))
        teacher_bw = fname_t_bw.map(lambda fname: util.ops.per_batch_op(fname, _decode_flo))

        self.set('training_data_teacher_flow_fw', teacher_fw)
        self.set('training_data_teacher_flow_bw', teacher_bw)

    def get_new_flo_path(self, base_path, orig_path, new_base_path, fw_str="_fw"):
        orig_path = orig_path.decode('utf-8')
        ext_split = orig_path.split('.')
        ext_split[-2] += fw_str
        ext_split[-1] = 'flo'
        ext_changed = '.'.join(ext_split)
        new_path = os.path.join(new_base_path, os.path.relpath(ext_changed, base_path))
        return new_path

def _get_flow_of_frame(fname):
    s = os.path.split(fname)
    frame_name = os.path.splitext(s[-1])[-2]
    scene_name = frame_name.split("_")[-2]
    path = os.path.join(s[-2], "%s_flow.flo"%scene_name)
    return path


def _decode_flo(abs_image_path):
    flo_string = tf.io.read_file(abs_image_path)
    width_string = tf.strings.substr(flo_string, 4, 4)
    height_string = tf.strings.substr(flo_string, 8, 4)
    width = tf.reshape(tf.io.decode_raw(width_string, out_type=tf.int32), [])
    height = tf.reshape(tf.io.decode_raw(height_string, out_type=tf.int32), [])

    value_flow = tf.strings.substr(flo_string, 12, height * width * 2 * 4)
    flow = tf.io.decode_raw(value_flow, out_type=tf.float32)

    return tf.reshape(flow, [height, width, 2])


def _decode_png(abs_image_path):
    image_string = tf.io.read_file(abs_image_path)
    image_decoded = tf.image.decode_image(image_string, channels=3, dtype=tf.float32)
    return image_decoded


def _decode_ppm(abs_image_path):
    IMG_SIZE = (384, 512)
    img = tf.numpy_function(read_ppm, [abs_image_path], tf.float32)
    img = tf.reshape(img, [IMG_SIZE[0], IMG_SIZE[1], 3])
    return img

