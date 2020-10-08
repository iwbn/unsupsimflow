from box import Box, BoxKeyError
import os
DEBUG = True

class ConfigBox(Box):
    def __init__(self, *args, **kwargs):
        super(ConfigBox, self).__init__(*args, **kwargs)

    def get_or_none(self, item):
        try:
            return self[item]
        except BoxKeyError:
            return None

    def get_or_zero(self, item):
        try:
            return self[item]
        except BoxKeyError:
            return 0

    def exist_or(self, *args):
        for k in args:
            try:
                tmp = self[k]
                return True
            except BoxKeyError:
                continue
        return False

    def save_or_load(self):
        yaml_path = os.path.join(self.ckpt_path, 'params.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                new_params = Box.from_yaml(f.read())
                self.update(new_params)
                print(self)
            print("config loaded (\"%s\")" % yaml_path)
        else:
            os.makedirs(self.ckpt_path, exist_ok=True)
            self.to_yaml(yaml_path)
            print("config saved (\"%s\")" % yaml_path)

    def copy(self):
        return ConfigBox(super(Box, self).copy())

if __name__ == "__main__":
    a = ConfigBox()
    a['a'] = 1
    print(a.a)
    print(a.b)
    print(a.exist_or("b", "c"))