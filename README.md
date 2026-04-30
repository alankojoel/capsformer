# Deep Beamforming With Capsule Networks
This repository includes the source code used in the paper Deep Beamforming With Capsule Networks.

## Abstract
Deep learning–based beamforming has advanced beyond classical techniques by relaxing restrictive assumptions such as prior knowledge of the system’s statistical properties and/or signal steering vectors. The existing approaches are however typically focused on single-signal beamforming, and some require explicit estimates of the steering vector properties of the received signals. To address these limitations, we propose a deep learning approach, termed CapsFormer, which integrates steering vector property estimation and beamforming within a unified neural network architecture. Specifically, capsule networks are employed to first estimate the properties of the incoming signals’ steering vectors and subsequently condition the beamforming weight computation on these estimates. This design enables flexible multi-signal beamforming in a single forward pass, while also allowing the incorporation of externally provided steering vector estimates for more precise conditioning. Simulation results demonstrate near-optimal beamforming performance for multiple signals without prior steering vector knowledge. The latter is remarkable as an indication that CapsFormer provides also a novel robust beamforming approach, which differs from the traditional uncertainty region-based robust approaches. 

## Overview
This repository has the following python scripts:
- `main.py` is used to run the simulations including data generation, model training and evaluation
- `src.data_generator.py` handles the training and validation data generation
- `src.training.py` handles the training procedure
- `src.evaluation.py` handles the evaluation procedure
- `src.models.py` defines the model achitectures
- `src.utils.py` defines utility functions

Additionally:
- File `config.yaml` includes the configuration parameters