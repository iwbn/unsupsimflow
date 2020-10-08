import tensorflow as tf
from .flyingchairs import FlyingChairs, FlyingChairsDD
from .sintel import Sintel, SintelDD
import data.data_util as data_util

def input_fn(mode, params):
    if mode == "train":
        if params.dataset.lower() == "flyingchairs":
            if params.exist_or("dd_weight"):
                data = FlyingChairsDD(params)
            else:
                data = FlyingChairs(params)
            data.initialize()
            im = data.get("training_data")
            flow = data.get("training_flow")

            if params.exist_or("dd_weight"):
                tflow_fw = data.get("training_data_teacher_flow_fw")
                tflow_bw = data.get("training_data_teacher_flow_bw")
                dataset = tf.data.Dataset.zip((im, tflow_fw, tflow_bw, flow))
                keys = [["im", "teacher_flow_fw_full", "teacher_flow_bw_full"], ["flow"]]
                flow_indices = [1,2,3]
            else:
                dataset = tf.data.Dataset.zip((im, flow))
                keys = [["im"], ["flow"]]
                flow_indices = [1]

        elif params.dataset.lower() == "sintel":
            if params.exist_or("dd_weight"):
                data = SintelDD(params)
            else:
                data = Sintel(params)
            data.initialize()

            clean_im = data.get("training_clean")
            clean_flow = data.get("training_flow")
            final_im = data.get("training_final")
            final_flow = data.get("training_flow")

            if params.exist_or("dd_weight"):
                clean_tflow_fw = data.get("training_clean_teacher_flow_fw")
                final_tflow_fw = data.get("training_final_teacher_flow_fw")
                clean_tflow_bw = data.get("training_clean_teacher_flow_bw")
                final_tflow_bw = data.get("training_final_teacher_flow_bw")
                dataset_clean = tf.data.Dataset.zip((clean_im, clean_tflow_fw, clean_tflow_bw, clean_flow))
                dataset_final = tf.data.Dataset.zip((final_im, final_tflow_fw, final_tflow_bw, final_flow))
                dataset = dataset_clean.concatenate(dataset_final)
                keys = [["im", "teacher_flow_fw_full", "teacher_flow_bw_full"], ["flow"]]
                flow_indices = [1,2,3]
            else:
                dataset_clean = tf.data.Dataset.zip((clean_im, clean_flow))
                dataset_final = tf.data.Dataset.zip((final_im, final_flow))

                dataset = dataset_clean.concatenate(dataset_final)
                keys = [["im"], ["flow"]]
                flow_indices = [1]

        else:
            raise NotImplementedError

        augmented = augment(dataset, flow_indices, params)
        labeled = data_util.convert_feat_label_to_dict(augmented, keys)

    else:
        if params.dataset.lower() == "flyingchairs":
            data = FlyingChairs(params)
            data.initialize()
            im = data.get("testing_data")
            flow = data.get("testing_flow")
            dataset = tf.data.Dataset.zip((im, flow))
            keys = [["im"], ["flow"]]
            flow_indices = [1]

        elif params.dataset.lower() == "sintel":
            data = Sintel(params)
            data.initialize()

            final_im = data.get("training_final")
            final_flow = data.get("training_flow")

            dataset_final = tf.data.Dataset.zip((final_im, final_flow))

            flow = data.get("training_flow")
            dataset = dataset_final
            keys = [["im"], ["flow"]]
            flow_indices = [1]
        else:
            raise NotImplementedError

        labeled = data_util.convert_feat_label_to_dict(dataset, keys)
    return labeled

def augment(dataset, flow_indices, params):
    dataset = data_util.rand_resize_multiple(dataset, flow_indices=flow_indices)
    dataset = data_util.rand_crop_multiple(dataset, params.input_size)
    dataset = data_util.rand_flip_multiple(dataset, flow_indices=flow_indices)
    dataset = data_util.rand_channel_swap_multiple(dataset, flow_indices=flow_indices)
    dataset = data_util.distort_color_multiple(dataset, skip_indices=flow_indices)
    return dataset
