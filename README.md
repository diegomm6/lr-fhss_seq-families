
```src.base``` contains common functions and classes

```src.families``` contains sequence families modules

```src.models.simulation.py``` defines the simulator module for per slot collisions

```src.models.simulationCR.py``` defines the simulator module for per packet collisions, considering the coding rate CR

```*.csv``` collision rate data files

```plots.ipynb``` is for ploting the data obtained in simulations

```testFamilies.py``` is for testing multiple families a single network size using parallel computation

```testNetsize.py``` is for testing multiple values for the network size for a sinlge family, using parallel computation

```testNetsizeCR.py``` variation the previous file for per packet collision

```test.ipynb``` is for developing and testing methods, messy file


# New model implemented

- Support processors limiting the number of parallel decoding at the dateway 
