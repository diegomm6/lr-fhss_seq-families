from abc import ABC
from src.base.LoRaTransmission import LoRaTransmission

class AbstractEvent(ABC):

    def __init__(self, time, name) -> None:
        super().__init__()
        self._time = time
        self._name = name


class StartEvent(AbstractEvent):

    def __init__(self, time, name, transmission) -> None:
        super().__init__(time, name)
        self._transmission : LoRaTransmission = transmission


class CollisionEvent(AbstractEvent):

    def __init__(self, time, name, ocw, obw) -> None:
        super().__init__(time, name)
        self._ocw = ocw
        self._obw = obw


class EndEvent(AbstractEvent):

    def __init__(self, time, name, transmission) -> None:
        super().__init__(time, name)
        self._transmission : LoRaTransmission = transmission
    