from abc import ABC, abstractmethod

class FHSfamily(ABC):

    def __init__(self, q) -> None:
        super().__init__()
        self.q = q
    
    @abstractmethod
    def get_family():
        pass
