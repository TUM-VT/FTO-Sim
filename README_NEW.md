# FTO-Sim: An Open-Source Framework for Cooperative Perception Analysis

## Introduction

**FTO-Sim** is a comprehensive open-source simulation framework designed to analyze cooperative perception systems in urban environments through integration of microscopic traffic simulation and advanced ray tracing techniques. The framework enables researchers and practitioners to evaluate the visibility coverage and detection performance of Floating Traffic Observers (FTOs), including Floating Car Observers (FCOs) and Floating Bike Observers (FBOs), in various traffic scenarios.

The framework extends the traditional Floating Car Observer (FCO) concept, originally developed for traffic monitoring using extended Floating Car Data (xFCD), to incorporate multiple observer types and evaluate their effectiveness in detecting surrounding traffic, in particular vulnerable road users (VRUs). By combining SUMO traffic simulation with sophisticated occlusion modeling (see [Ray Tracing Methodology](#6-ray-tracing-methodology)), FTO-Sim provides detailed insights into the spatial and temporal characteristics of cooperative perception systems.

FTO-Sim addresses critical challenges in the deployment of cooperative intelligent transportation systems (C-ITS) by quantifying how physical occlusions from buildings, vegetation, parked vehicles (static occlusion) and other road users (dynamic occlusion) affect the field of view of observer vehicles. The framework's modular architecture (see [Framework Architecture](#5-framework-architecture)) supports multiple evaluation approaches, including [spatial visibility analysis](#8-spatial-visibility-analysis) for infrastructure-wide assessment and [VRU-specific detection metrics](#9-vru-specific-detection-analysis) for more targeted safety evaluation.

## About this Documentation

This documentation provides a complete guide to understanding, installing, configuring, and using the FTO-Sim framework. It is structured to serve both as a comprehensive reference for researchers seeking methodological details and as a practical manual for users implementing the framework in their own studies.

The documentation is organized to first cover essential information such as [citation requirements](#1-citation), [installation procedures](#2-installation-and-prerequisites), and basic [configuration](#3-configuration). Subsequently, it presents the [framework's architecture](#5-framework-architecture) and internal workflows, followed by in-depth explanations of the methodological foundations underlying the [ray tracing algorithm](#6-ray-tracing-methodology) and [evaluation metrics](#8-spatial-visibility-analysis). Finally, [simulation examples](#12-simulation-examples) that replicate previous work demonstrate how to apply FTO-Sim.

The [configuration](#3-configuration) and [usage](#4-usage) sections provide step-by-step instructions for framework operation, while methodological sections explain the scientific foundations and assumptions of implemented algorithms, which are described in more detail in the accompanying publications (see [Primary References](#11-primary-references)).

This documentation assumes basic familiarity with traffic simulation concepts and Python programming. Users should have a working knowledge of SUMO (Simulation of Urban MObility) and understand fundamental concepts of coordinate systems and spatial data processing.

---

## Table of Contents

1. [Citation](#1-citation)
   - 1.1 [Primary References](#11-primary-references)
   - 1.2 [Secondary References](#12-secondary-references)

2. [Installation and Prerequisites](#2-installation-and-prerequisites)
   - 2.1 [System Requirements](#21-system-requirements)
   - 2.2 [SUMO Installation](#22-sumo-installation)
   - 2.3 [Python Environment Setup](#23-python-environment-setup)
   - 2.4 [Creating a Virtual Environment](#24-creating-a-virtual-environment)
   - 2.5 [Activating the Virtual Environment](#25-activating-the-virtual-environment)
   - 2.6 [Installing Required Packages](#26-installing-required-packages)
   - 2.7 [Optional GPU Acceleration](#27-optional-gpu-acceleration)
   - 2.8 [Verifying the Installation](#28-verifying-the-installation)

3. [Configuration](#3-configuration)
   - 3.1 [Configuration File Overview](#31-configuration-file-overview)
   - 3.2 [Simulation Identification Settings](#32-simulation-identification-settings)
   - 3.3 [Performance Optimization Settings](#33-performance-optimization-settings)
   - 3.4 [Path Configuration](#34-path-configuration)
   - 3.5 [Geographic Bounding Box Settings](#35-geographic-bounding-box-settings)
   - 3.6 [Simulation Warm-up Settings](#36-simulation-warm-up-settings)
   - 3.7 [Observer Penetration Rate Settings](#37-observer-penetration-rate-settings)
   - 3.8 [Ray Tracing Parameter Settings](#38-ray-tracing-parameter-settings)
   - 3.9 [Visualization Settings](#39-visualization-settings)
   - 3.10 [Data Collection Settings](#310-data-collection-settings)
   - 3.11 [Advanced Configuration Options](#311-advanced-configuration-options)

4. [Usage](#4-usage)
   - 4.1 [Basic Workflow](#41-basic-workflow)
   - 4.2 [Simulation Mode](#42-simulation-mode)
   - 4.3 [Visualization Mode](#43-visualization-mode)
   - 4.4 [Debugging Mode](#44-debugging-mode)
   - 4.5 [Saving Mode](#45-saving-mode)
   - 4.6 [Running Evaluation Scripts](#46-running-evaluation-scripts)
   - 4.7 [Common Workflows and Use Cases](#47-common-workflows-and-use-cases)

5. [Framework Architecture](#5-framework-architecture)
   - 5.1 [Overview](#51-overview)
   - 5.2 [Modular Design Philosophy](#52-modular-design-philosophy)
   - 5.3 [Configuration and Initialization](#53-configuration-and-initialization)
   - 5.4 [Input Data Processing](#54-input-data-processing)
   - 5.5 [Coordinate Transformation](#55-coordinate-transformation)
   - 5.6 [Ray Tracing Engine](#56-ray-tracing-engine)
   - 5.7 [Data Logging System](#57-data-logging-system)
   - 5.8 [Evaluation and Post-Processing](#58-evaluation-and-post-processing)
   - 5.9 [Workflow Integration](#59-workflow-integration)

6. [Ray Tracing Methodology](#6-ray-tracing-methodology)
   - 6.1 [Theoretical Foundation](#61-theoretical-foundation)
   - 6.2 [Initialization and Binning Map Setup](#62-initialization-and-binning-map-setup)
   - 6.3 [Observer Assignment Process](#63-observer-assignment-process)
   - 6.4 [Ray Generation and Distribution](#64-ray-generation-and-distribution)
   - 6.5 [Occlusion Detection Algorithm](#65-occlusion-detection-algorithm)
   - 6.6 [Ray Intersection Calculations](#66-ray-intersection-calculations)
   - 6.7 [Visibility Polygon Construction](#67-visibility-polygon-construction)
   - 6.8 [Performance Optimization Strategies](#68-performance-optimization-strategies)
   - 6.9 [Visualization Capabilities](#69-visualization-capabilities)

7. [Data Collection and Logging](#7-data-collection-and-logging)
   - 7.1 [Data Logging Architecture](#71-data-logging-architecture)
   - 7.2 [Output Directory Structure](#72-output-directory-structure)
   - 7.3 [Core Simulation Logs](#73-core-simulation-logs)
   - 7.4 [Spatial Visibility Data](#74-spatial-visibility-data)
   - 7.5 [Real-Time Data Collection Process](#75-real-time-data-collection-process)
   - 7.6 [Data Integrity and Validation](#76-data-integrity-and-validation)
   - 7.7 [Performance Considerations](#77-performance-considerations)

8. [Spatial Visibility Analysis](#8-spatial-visibility-analysis)
   - 8.1 [Conceptual Foundation](#81-conceptual-foundation)
   - 8.2 [Methodology Overview](#82-methodology-overview)
   - 8.3 [Relative Visibility Metric](#83-relative-visibility-metric)
   - 8.4 [Level of Visibility (LoV) Metric](#84-level-of-visibility-lov-metric)
   - 8.5 [LoV Classification Methodology](#85-lov-classification-methodology)
   - 8.6 [Heatmap Generation and Visualization](#86-heatmap-generation-and-visualization)
   - 8.7 [Interpretation Guidelines](#87-interpretation-guidelines)
   - 8.8 [Limitations and Considerations](#88-limitations-and-considerations)
   - 8.9 [Running Spatial Visibility Evaluation](#89-running-spatial-visibility-evaluation)

9. [VRU-Specific Detection Analysis](#9-vru-specific-detection-analysis)
   - 9.1 [Motivation and Objective](#91-motivation-and-objective)
   - 9.2 [Detection Event Tracking](#92-detection-event-tracking)
   - 9.3 [Spatio-Temporal Detection Rates](#93-spatio-temporal-detection-rates)
   - 9.4 [Critical Interaction Areas](#94-critical-interaction-areas)
   - 9.5 [Multi-Level Aggregation](#95-multi-level-aggregation)
   - 9.6 [Trajectory Visualization](#96-trajectory-visualization)
   - 9.7 [Defining Critical Areas in SUMO](#97-defining-critical-areas-in-sumo)
   - 9.8 [Interpretation and Application](#98-interpretation-and-application)
   - 9.9 [Running VRU-Specific Detection Evaluation](#99-running-vru-specific-detection-evaluation)

10. [Conflict Analysis and Safety Metrics](#10-conflict-analysis-and-safety-metrics)
    - 10.1 [SUMO Surrogate Safety Measures](#101-sumo-surrogate-safety-measures)
    - 10.2 [Time-to-Collision (TTC)](#102-time-to-collision-ttc)
    - 10.3 [Post-Encroachment Time (PET)](#103-post-encroachment-time-pet)
    - 10.4 [Deceleration Rate to Avoid Crash (DRAC)](#104-deceleration-rate-to-avoid-crash-drac)
    - 10.5 [Integration with Detection Data](#105-integration-with-detection-data)
    - 10.6 [Limitations of Conflict Detection](#106-limitations-of-conflict-detection)

11. [Advanced Topics](#11-advanced-topics)
    - 11.1 [Custom Observer Types](#111-custom-observer-types)
    - 11.2 [Extending Evaluation Metrics](#112-extending-evaluation-metrics)
    - 11.3 [Large-Scale Simulations](#113-large-scale-simulations)
    - 11.4 [Integration with External Tools](#114-integration-with-external-tools)
    - 11.5 [Parameter Sensitivity Analysis](#115-parameter-sensitivity-analysis)

12. [Simulation Examples](#12-simulation-examples)
    - 12.1 [Overview of Included Examples](#121-overview-of-included-examples)
    - 12.2 [Example 1: Spatial Visibility Analysis (Ilic et al., TRB 2025)](#122-example-1-spatial-visibility-analysis-ilic-et-al-trb-2025)
    - 12.3 [Example 2: VRU-Specific Detection (Ilic et al., TRA 2026)](#123-example-2-vru-specific-detection-ilic-et-al-tra-2026)
    - 12.4 [Creating Custom Simulation Scenarios](#124-creating-custom-simulation-scenarios)
    - 12.5 [Best Practices for Scenario Development](#125-best-practices-for-scenario-development)

13. [Troubleshooting and FAQ](#13-troubleshooting-and-faq)
    - 13.1 [Common Installation Issues](#131-common-installation-issues)
    - 13.2 [SUMO Integration Problems](#132-sumo-integration-problems)
    - 13.3 [Performance Issues](#133-performance-issues)
    - 13.4 [Visualization Problems](#134-visualization-problems)
    - 13.5 [Data Output Issues](#135-data-output-issues)

14. [Contributing and Development](#14-contributing-and-development)
    - 14.1 [Contributing Guidelines](#141-contributing-guidelines)
    - 14.2 [Code Structure](#142-code-structure)
    - 14.3 [Testing](#143-testing)
    - 14.4 [Documentation Standards](#144-documentation-standards)

15. [License and Acknowledgments](#15-license-and-acknowledgments)
    - 15.1 [License Information](#151-license-information)
    - 15.2 [Acknowledgments](#152-acknowledgments)
    - 15.3 [Third-Party Dependencies](#153-third-party-dependencies)

---

## 1. Citation

When using FTO-Sim in your research, please cite the appropriate references to acknowledge the framework's development and methodological foundations.

### 1.1 Primary References

The following publications present different versions of the FTO-Sim framework and provide detailed methodological explanations of the implemented algorithms, theoretical foundations, and evaluation metrics.

#### ETRR Journal Paper (FTO-Sim Version 2)

* **Ilic, M., et al.** (2025). "FTO-Sim: An Open-Source Framework for Evaluating Cooperative Perception in Urban Areas." *European Transport Research Review*. *(Under review after first revisions)*

This paper presents the current version of FTO-Sim (Version 2) with comprehensive methodological foundations and explanations of all implemented evaluation metrics. The framework now includes both *spatial visibility analysis* and *VRU-specific detection* metrics for comprehensive assessment of cooperative perception systems. Furthermore, the paper introduces ready-to-use simulation examples included in the FTO-Sim repository, enabling rapid application of the framework for other researchers and practitioners.

#### TRB 2025 Conference Paper (FTO-Sim Version 1)

* [**Ilic, M., et al.**](https://www.researchgate.net/publication/383272173_An_Open-Source_Framework_for_Evaluating_Cooperative_Perception_in_Urban_Areas) (2025). "An Open-Source Framework for Evaluating Cooperative Perception in Urban Areas." *Transportation Research Board 104th Annual Meeting*, Washington D.C., USA.

This paper introduces the initial version of FTO-Sim (Version 1) with the first implementation of spatial visibility analysis metrics. It presents the foundational occlusion modeling approach and demonstrates the framework's capability for analyzing relative visibility patterns and the Level of Visibility (LoV) metric, originally introduced by [Pechinger et al.](https://www.researchgate.net/publication/372952261_THRESHOLD_ANALYSIS_OF_STATIC_AND_DYNAMIC_OCCLUSION_IN_URBAN_AREAS_A_CONNECTED_AUTOMATED_VEHICLE_PERSPECTIVE) Through a small case study focused on a single intersection, the paper identifies opportunities for further calibration and refinement of the LoV metric.

### 1.2 Secondary References

The following publications demonstrate the application of FTO-Sim to specific case studies and research questions, showcasing the framework's capabilities in various traffic scenarios and evaluation contexts.

#### TRA 2026 Conference Paper

* **Ilic, M., et al.** (2026). "Evaluating VRU Detection Performance in Urban Environments using Cooperative Perception." *Transport Research Arena (TRA) 2026*, Budapest, Hungary. *(01.-04.06.26, accepted for oral presentation, publication of proceedings pending)*

This paper applies FTO-Sim's VRU-specific detection metrics to evaluate the effectiveness of cooperative perception systems in detecting vulnerable road users. Through a comparative case study examining different speed limit scenarios (30 km/h vs. 50 km/h), the paper demonstrates how infrastructure design and traffic management measures influence detection rates in critical interaction areas. The study showcases the framework's capability to assess spatio-temporal detection performance at multiple aggregation levels.

---

## 2. Installation and Prerequisites

### 2.1 System Requirements

### 2.2 SUMO Installation

### 2.3 Python Environment Setup

### 2.4 Creating a Virtual Environment

### 2.5 Activating the Virtual Environment

### 2.6 Installing Required Packages

### 2.7 Optional GPU Acceleration

### 2.8 Verifying the Installation

---

## 3. Configuration

### 3.1 Configuration File Overview

### 3.2 Simulation Identification Settings

### 3.3 Performance Optimization Settings

### 3.4 Path Configuration

### 3.5 Geographic Bounding Box Settings

### 3.6 Simulation Warm-up Settings

### 3.7 Observer Penetration Rate Settings

### 3.8 Ray Tracing Parameter Settings

### 3.9 Visualization Settings

### 3.10 Data Collection Settings

### 3.11 Advanced Configuration Options

---

## 4. Usage

### 4.1 Basic Workflow

### 4.2 Simulation Mode

### 4.3 Visualization Mode

### 4.4 Debugging Mode

### 4.5 Saving Mode

### 4.6 Running Evaluation Scripts

### 4.7 Common Workflows and Use Cases

---

## 5. Framework Architecture

### 5.1 Overview

### 5.2 Modular Design Philosophy

### 5.3 Configuration and Initialization

### 5.4 Input Data Processing

### 5.5 Coordinate Transformation

### 5.6 Ray Tracing Engine

### 5.7 Data Logging System

### 5.8 Evaluation and Post-Processing

### 5.9 Workflow Integration

---

## 6. Ray Tracing Methodology

### 6.1 Theoretical Foundation

### 6.2 Initialization and Binning Map Setup

### 6.3 Observer Assignment Process

### 6.4 Ray Generation and Distribution

### 6.5 Occlusion Detection Algorithm

### 6.6 Ray Intersection Calculations

### 6.7 Visibility Polygon Construction

### 6.8 Performance Optimization Strategies

### 6.9 Visualization Capabilities

---

## 7. Data Collection and Logging

### 7.1 Data Logging Architecture

### 7.2 Output Directory Structure

### 7.3 Core Simulation Logs

### 7.4 Spatial Visibility Data

### 7.5 Real-Time Data Collection Process

### 7.6 Data Integrity and Validation

### 7.7 Performance Considerations

---

## 8. Spatial Visibility Analysis

### 8.1 Conceptual Foundation

### 8.2 Methodology Overview

### 8.3 Relative Visibility Metric

### 8.4 Level of Visibility (LoV) Metric

### 8.5 LoV Classification Methodology

### 8.6 Heatmap Generation and Visualization

### 8.7 Interpretation Guidelines

### 8.8 Limitations and Considerations

### 8.9 Running Spatial Visibility Evaluation

---

## 9. VRU-Specific Detection Analysis

### 9.1 Motivation and Objective

### 9.2 Detection Event Tracking

### 9.3 Spatio-Temporal Detection Rates

### 9.4 Critical Interaction Areas

### 9.5 Multi-Level Aggregation

### 9.6 Trajectory Visualization

### 9.7 Defining Critical Areas in SUMO

### 9.8 Interpretation and Application

### 9.9 Running VRU-Specific Detection Evaluation

---

## 10. Conflict Analysis and Safety Metrics

### 10.1 SUMO Surrogate Safety Measures

### 10.2 Time-to-Collision (TTC)

### 10.3 Post-Encroachment Time (PET)

### 10.4 Deceleration Rate to Avoid Crash (DRAC)

### 10.5 Integration with Detection Data

### 10.6 Limitations of Conflict Detection

---

## 11. Advanced Topics

### 11.1 Custom Observer Types

### 11.2 Extending Evaluation Metrics

### 11.3 Large-Scale Simulations

### 11.4 Integration with External Tools

### 11.5 Parameter Sensitivity Analysis

---

## 12. Simulation Examples

### 12.1 Overview of Included Examples

### 12.2 Example 1: Spatial Visibility Analysis (Ilic et al., TRB 2025)

### 12.3 Example 2: VRU-Specific Detection (Ilic et al., TRA 2026)

### 12.4 Creating Custom Simulation Scenarios

### 12.5 Best Practices for Scenario Development

---

## 13. Troubleshooting and FAQ

### 13.1 Common Installation Issues

### 13.2 SUMO Integration Problems

### 13.3 Performance Issues

### 13.4 Visualization Problems

### 13.5 Data Output Issues

---

## 14. Contributing and Development

### 14.1 Contributing Guidelines

### 14.2 Code Structure

### 14.3 Testing

### 14.4 Documentation Standards

---

## 15. License and Acknowledgments

### 15.1 License Information

### 15.2 License

### 15.3 Third-Party Dependencies

---

*This documentation is maintained by the FTO-Sim development team. For questions, issues, or contributions, please visit the project repository.*
