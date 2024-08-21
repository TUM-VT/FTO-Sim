# FTO-Sim
**FTO-Sim** is an open-source simulation framework for Floating Traffic Observation (FTO). The FTO concept is adapted from the Floating Car Observer (FCO) method, that utilizes extended floating car data (xFCD) for traffic planning and traffic management purposes. Additionally, this simulation framework introduces further observer vehicle types, such as Floating Bike Observers (FBO).

For this, **FTO-Sim** is connected to a SUMO simulation and customized according to the user's needs. This ReadMe file serves as documentation for **FTO-Sim** and describes the single modules of the simulation framework as well as its customization and initialization.

## Table of Contents
1. [Citation](#citation)
2. [Features](#features)
3. [Installation / Prerequisites](#installation--prerequisites)
4. [Customization](#customization)
5. [Usage](#usage)

## Citation
When using **FTO-Sim**, please cite the following references:
* [Introduction of FTO-Sim](https://www.researchgate.net/publication/383272173_An_Open-Source_Framework_for_Evaluating_Cooperative_Perception_in_Urban_Areas) includes a detailed description of the general features of the simulation framework. Furthermore, a first application (visibility analysis) of the simulation framework is included to further calibrate the Level of Visibility (LoV) metric, originally introduced by [Pechinger et al.](https://www.researchgate.net/publication/372952261_THRESHOLD_ANALYSIS_OF_STATIC_AND_DYNAMIC_OCCLUSION_IN_URBAN_AREAS_A_CONNECTED_AUTOMATED_VEHICLE_PERSPECTIVE).

## Features
The following sub-chapters elaborate on the different modules and functionalities of the simulation framework, which are summarized in the following figure.

![Overview of the FTO-Sim Framework Architecture](readme_images/framework_features.png)

### Input Data

...

### Configuration Settings

...

### Ray Tracing

...

### Applications

...

## Installation / Prerequisites

Create venv:
```
python -m venv venv
```

Initalizie / Activate venv:
```
.\venv\Scripts\activate
```

Installing required packages:
```
pip install -r requirements.txt
```

## Customization

Introduce all customization modules of the simulation framework (similar to TRB paper)

## Usage

How to rum the simulation framework (relate to customization for different 'use modes', e.g. Visualization, ManualForwarding, etc.)