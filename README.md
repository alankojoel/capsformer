# Deep Beamforming With Capsule Networks
<!-- This repository includes the source code used in the paper Deep Beamforming With Capsule Networks. -->

## Abstract
Deep learning-based beamforming has improved upon classical techniques while relaxing restrictive assumptions such as prior knowledge of the statistical properties of signal directions of arrivals (DOAs). However, many of the proposed solutions focus on single signal beamforming and some require estimates of the DOAs. To overcome these limitations, we propose a deep learning approach, called CapsFormer, that integrates source localization and beamforming into a single neural network (NN) architecture. Specifically, we utilize capsule networks to first estimate the DOAs of incoming signals and then condition the beamforming weight computation on the estimated DOAs. This enables flexible multi-signal beamforming within a single pass while the model design also supports incorporating external DOA estimates for precise location conditioning. Simulation results showcase near-optimal beamforming performance for multiple signals without prior DOA knowledge, as well for single signal in the presence of interferers and DOA mismatch. 

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
