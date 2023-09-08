import os
import datetime


class Logger:
    def __init__(self, path, filename):
        # path = os.path.join('./logs/', path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.filepath = os.path.join(path, filename)
        return

    def log(self, dic: dict):
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                f.write(';'.join(dic) + '\n')
        with open(self.filepath, 'a') as f:
            f.write(';'.join(map(lambda x: str(x), dic.values())) + '\n')
