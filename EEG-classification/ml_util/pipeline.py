from typing import Callable, Any

class Pipeline:
    def __init__(self):
        self.functions = []

    def add(self, f: Callable["...", Any]):
        self.functions.append(f)

    def __call__(self, x, permarg=None):
        tmp = x
        for f in self.functions:
            tmp = f(x, permarg)
        return tmp
