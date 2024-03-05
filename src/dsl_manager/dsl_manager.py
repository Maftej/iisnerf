from abc import ABC, abstractmethod


class DslManager(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run_all_scenarios(self):
        pass

    @abstractmethod
    def run_single_scenario(self, scenario):
        pass
