from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def activate(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass
