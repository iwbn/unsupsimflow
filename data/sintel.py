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

BASE_PATH = pf.get_path('SintelBasePath')
BASE_PATH_FULLSET = pf.get_path('SintelFullsetBasePath')
FULLSET_NUM_PNG = 21312
TRAINING_REL = "training"
TEST_REL = "test"
DATA_TYPES = ['final', 'flow', 'albedo', 'occlusions', 'clean', 'flow_viz', 'invalid']


class Sintel(Data):
    def __init__(self, params):
        Data.__init__(self, "Sintel")
        self.params = params

    def prepare(self):
        train_scenes = os.listdir(os.path.join(BASE_PATH, TRAINING_REL, 'final'))
        test_scenes = os.listdir(os.path.join(BASE_PATH, TEST_REL, 'final'))

        all_scenes = train_scenes + test_scenes
        num_scenes = len(all_scenes)

        fname_list_dict = {}

        training_scenes = _get_all_scenes_in_rel(BASE_PATH, TRAINING_REL)
        testing_scenes = _get_all_scenes_in_rel(BASE_PATH, TEST_REL)



        seq_len = 2
        print(training_scenes)
        nframes_training = {name: len(training_scenes['final'][name]) - seq_len + 1
                               for name in training_scenes['final'].keys()}
        nframes_testing = {name: len(testing_scenes['final'][name]) - seq_len + 1
                              for name in testing_scenes['final'].keys()}

        fnames_fullset = [[os.path.join(BASE_PATH_FULLSET, "%05d.png" % (j + i+ 1)) for j in range(seq_len)]
                          for i in range(0, FULLSET_NUM_PNG - seq_len + 1)]

        def _get_fname_datasets(nframes, scenes):
            datasets = {}
            for scenetype in scenes.keys():
                x = []
                for scenename in nframes.keys():
                    for fidx in range(nframes[scenename]):
                        if len(scenes[scenetype][scenename]) == nframes[scenename] + seq_len - 1:
                            fnames = [scenes[scenetype][scenename][fidx + i] for i in range(seq_len)]
                        else:
                            fnames = [scenes[scenetype][scenename][fidx + i] for i in range(seq_len - 1)]
                        x.append(fnames)

                dataset = tf.data.Dataset.from_tensor_slices(x)
                datasets[scenetype] = dataset

            return datasets


        def _get_img_or_flow(fname_dataset, is_flow=False):
            if is_flow:
                dataset = fname_dataset.map(lambda fname: util.ops.per_batch_op(fname, _decode_flo))
            else:
                dataset = fname_dataset.map(lambda fname: util.ops.per_batch_op(fname, _decode_png))
            return dataset

        def get_datasets(fname_datasets):
            loaded_datasets = {}
            for k, dataset in fname_datasets.items():
                if k == "flow":
                    loaded_datasets[k] = _get_img_or_flow(dataset, is_flow=True)
                else:
                    loaded_datasets[k] = _get_img_or_flow(dataset)

            return loaded_datasets

        tnames = _get_fname_datasets(nframes_training, training_scenes)
        tsnames = _get_fname_datasets(nframes_testing, testing_scenes)
        fullnames = tf.data.Dataset.from_tensor_slices(fnames_fullset)

        tdata = get_datasets(tnames)
        tsdata = get_datasets(tsnames)
        fulldata = get_datasets({'full': fullnames})

        for k, v in tnames.items():
            self.set("training_%s_fname" % k, v)


        for k, v in tsnames.items():
            self.set("testing_%s_fname" % k, v)

        for k, v in tdata.items():
            self.set("training_%s" % k, v)

        for k, v in tsdata.items():
            self.set("testing_%s" % k, v)

        self.set("movie_full_fname", fullnames)

        for k, v in fulldata.items():
            self.set("movie_%s" % k, v)

    def get_relative_path_of(self, abspath):
        return os.path.relpath(abspath, BASE_PATH)

class SintelDD(Sintel):
    def prepare(self):
        Sintel.prepare(self)
        fname_clean = self.get("training_clean_fname")
        fname_final = self.get("training_final_fname")
        #fname_full = self.get("movie_full_fname")

        ## load teacher flow
        teacher_flow_base_path = self.params.teacher_flow_path

        _get_mpi_new = lambda x: tf.numpy_function(
            lambda orig_path: self.get_new_flo_path(BASE_PATH, orig_path, teacher_flow_base_path),
            [x], tf.string
        )
        _get_mpi_new_bw = lambda x: tf.numpy_function(
            lambda orig_path: self.get_new_flo_path(BASE_PATH, orig_path, teacher_flow_base_path, "_bw"),
            [x], tf.string
        )
        #_get_fullset_new = lambda orig_path: self.get_new_flo_path(BASE_PATH_FULLSET, orig_path, teacher_flow_base_path)

        fname_clean_t = fname_clean.map(lambda fnames: util.ops.per_batch_op(fnames[:-1], _get_mpi_new, tf.string))
        fname_final_t = fname_final.map(lambda fnames: util.ops.per_batch_op(fnames[:-1], _get_mpi_new, tf.string))
        fname_clean_t_bw = fname_clean.map(lambda fnames: util.ops.per_batch_op(fnames[:-1], _get_mpi_new_bw, tf.string))
        fname_final_t_bw = fname_final.map(lambda fnames: util.ops.per_batch_op(fnames[:-1], _get_mpi_new_bw, tf.string))

        self.set("training_clean_teacher_flow_fname", fname_clean_t)
        self.set("training_final_teacher_flow_fname", fname_final_t)
        self.set("training_clean_teacher_flow_bw_fname", fname_clean_t_bw)
        self.set("training_final_teacher_flow_bw_fname", fname_final_t_bw)

        def _get_img_or_flow(fname_dataset, is_flow=False):
            if is_flow:
                dataset = fname_dataset.map(lambda fname: util.ops.per_batch_op(fname, _decode_flo))
            else:
                dataset = fname_dataset.map(lambda fname: util.ops.per_batch_op(fname, _decode_png))
            return dataset

        def get_datasets(fname_datasets):
            loaded_datasets = {}
            for k, dataset in fname_datasets.items():
                loaded_datasets[k] = _get_img_or_flow(dataset, is_flow=True)

            return loaded_datasets

        teacher_datasets = get_datasets({'training_clean_teacher_flow': fname_clean_t,
                                         'training_final_teacher_flow': fname_final_t,
                                         'training_clean_teacher_flow_bw': fname_clean_t_bw,
                                         'training_final_teacher_flow_bw': fname_final_t_bw,
                                         })

        self.set('training_clean_teacher_flow_fw', teacher_datasets['training_clean_teacher_flow'])
        self.set('training_final_teacher_flow_fw', teacher_datasets['training_final_teacher_flow'])
        self.set('training_clean_teacher_flow_bw', teacher_datasets['training_clean_teacher_flow_bw'])
        self.set('training_final_teacher_flow_bw', teacher_datasets['training_final_teacher_flow_bw'])

    def get_new_flo_path(self, base_path, orig_path, new_base_path, fw_str="_fw"):
        orig_path = orig_path.decode('utf-8')
        ext_split = orig_path.split('.')
        ext_split[-2] += fw_str
        ext_split[-1] = 'flo'
        ext_changed = '.'.join(ext_split)
        new_path = os.path.join(new_base_path, os.path.relpath(ext_changed, base_path))
        return new_path



def _get_all_scenes_in_rel(base_path, rel):
    scenetypes = [filename for filename in os.listdir(os.path.join(base_path, rel))
                     if os.path.isdir(os.path.join(base_path, rel, filename)) and not filename.startswith(".")]
    scenes = {}
    for typename in scenetypes:
        scenenames = [filename for filename in os.listdir(os.path.join(base_path, rel, typename))
                      if os.path.isdir(os.path.join(base_path, rel, typename, filename))]
        scenes[typename] = {}
        for name in scenenames:
            scenes[typename][name] = _get_file_list(os.path.join(base_path, rel, typename, name), get_relative=False)
    return scenes

def _get_file_list(abs_image_dir, extension="", get_relative=False):
    filenames = [filename for filename in os.listdir(abs_image_dir)
                     if filename.lower().endswith(extension.lower())]
    filenames.sort()
    if get_relative:
        return filenames
    abs_filenames = [os.path.join(abs_image_dir, filename) for filename in filenames]
    return abs_filenames


def _get_flow_of_frame(fname):
    s = os.path.split(fname)
    frame_name = os.path.splitext(s[-1])[-2]
    scene_name = frame_name.split("_")[-2]
    path = os.path.join(s[-2], "%s_flow.flo"%scene_name)
    return path


def _get_fake_flow_of_frame(fake_flow_path, tail, fname):
    s = os.path.split(fname)
    scene_name = s[-2].split("/")[-1]
    frame_name = os.path.splitext(s[-1])[-2]
    frame_name = frame_name.replace("img1", 'flow')
    path = os.path.join(fake_flow_path, scene_name, frame_name + tail) + ".flo"
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
    image_decoded = tf.io.decode_image(image_string, channels=3, dtype=tf.float32)
    s = tf.shape(image_decoded)
    return tf.reshape(image_decoded, [s[0], s[1], 3])


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
