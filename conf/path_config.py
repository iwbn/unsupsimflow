from box import Box

paths = Box()
paths.FlyingChairsBasePath = "/mnt/data/wbim/FlyingChairs/FlyingChairs_release/data/"
paths.FlyingChairsMetaFilePath = "/mnt/data/wbim/FlyingChairs/FlyingChairs_train_val.txt"
paths.SintelBasePath = "/mnt/data/wbim/Sintel"
paths.SintelFullsetBasePath = "/mnt/data/wbim/Sintel/png"
paths.KITTIBasePath = "/mnt/data/wbim/KITTI/"

def get_path(path):
    return paths[path]