from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def activate(self, *args, **kwargs):
        pass

    @abstractmethod
    def derivative(self, x, **kwargs):
        pass
