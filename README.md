
LR-FHSS network simulation for collision analysis. Two models available:

- ```./src/models/LoRaNetworkLite.py``` has unlimited decoding capacity, basic model, deprecated.

- ```./src/models/LoRaNetwork.py``` has limited decoding capacity depending on the available Processors in the Gateway. Support for count-based and power-based collision analysis. Support for dynamic doppler.
