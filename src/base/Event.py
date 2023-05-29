from abc import ABC
from src.base.LoRaTransmission import LoRaTransmission

class AbstractEvent(ABC):

    def __init__(self, time, name) -> None:
        super().__init__()
        self._time = time
        self._name = name
    
    def __lt__(self, other) -> bool:
        return self._time < other._time
    
    def __str__(self) -> str:
        return f"{self._name} at time {self._time}"
    

class StartEvent(AbstractEvent):

    def __init__(self, time, name, transmission) -> None:
        super().__init__(time, name)
        self._transmission : LoRaTransmission = transmission

    def __str__(self) -> str:
        return f"start transmission {self._transmission.id} at time {self._time}"


class CollisionEvent(AbstractEvent):

    def __init__(self, time, name, ocw, obw) -> None:
        super().__init__(time, name)
        self._ocw = ocw
        self._obw = obw

    def __str__(self) -> str:
        return f"collision in (ocw={self._ocw}, obw={self._obw}) at time {self._time}"


class EndEvent(AbstractEvent):

    def __init__(self, time, name, transmission) -> None:
        super().__init__(time, name)
        self._transmission : LoRaTransmission = transmission

    def __str__(self) -> str:
        return f"ends transmission {self._transmission.id} at time {self._time}"
    