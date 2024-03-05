from abc import ABC, abstractmethod


class AbstractTrainingManager(ABC):
    @abstractmethod
    def train(self, single_scenario):
        pass
