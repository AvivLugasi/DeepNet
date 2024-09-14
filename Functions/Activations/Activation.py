from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def activate(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.activate(*args, **kwargs)

    @abstractmethod
    def derivative(self, x, **kwargs):
        pass
