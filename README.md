# TSN Scheduling Algorithm for Real-Time Applications in a Heterogeneous Network

This repository documents the work presented in the Master's thesis **"TSN Scheduling Algorithm for Real-Time Applications in a Heterogeneous Network"**, focused on the design and evaluation of scheduling algorithms for IEEE 802.1Qbv Time-Aware Shaper (TAS).

---

## Overview

Time-Sensitive Networking (TSN) enables deterministic Ethernet communication for real-time applications with strict latency and jitter constraints. This work addresses the problem of **scheduling time-sensitive (TS) flows** in **heterogeneous networks**, where links differ in throughput, processing modes, and transmission characteristics.

The core contribution is a **TAS scheduling framework** that supports:

* Heterogeneous network interfaces
* Multi-period TS flows
* Scheduling with and without data-plane reconfiguration
* End-to-end delay and jitter optimization

The thesis was developed within the **NextGeneration UNICO5G – TIMING project**.

---

## Problem Statement

Given a time-sensitive network with an existing scheduling plan:

1. **Scheduling without reconfiguration**
   Accept a new TS request if feasible, without modifying the incumbent data plane.

2. **Scheduling with reconfiguration**
   Accept a new TS request while producing a new scheduling plan that minimizes changes with respect to the incumbent one.

Challenges addressed include:

* Heterogeneous throughput and processing modes (express vs store-and-forward)
* Transmission modes (simplex, half-duplex, full-duplex)
* Propagation delays and unavailable time slots
* Pipeline dependencies between consecutive interfaces

---

## IEEE 802.1Qbv Time-Aware Shaper (TAS)

The TAS divides time into repeating **Super Frames (SF)** composed of discrete time slots. For each network interface, the scheduler:

* Discretizes the SF into atomic time slots
* Assigns scheduling windows to TS flows
* Guarantees guard bands and synchronization
* Respects delay and jitter constraints
* Solves a global optimization problem

---

## Models

### ILP Model

Two Integer Linear Programming (ILP) formulations are proposed:

* **Scheduling without reconfiguration**
  Objective: minimize end-to-end jitter and delay.

* **Scheduling with reconfiguration**
  Objective: minimize a weighted combination of:

  * Maximum delay
  * Maximum jitter
  * Number of data-plane changes

The models are inspired by the **Job Shop Scheduling Problem (JSSP)**:

* Network interfaces → machines
* TS flow iterations → jobs
* Scheduling windows → tasks

---

## Heuristic Algorithm

To overcome ILP scalability limits, a heuristic approach is introduced:

1. **Constructive phase**

   * Finds locally optimal starting time slots
   * Allocates scheduling windows

2. **Local Search (LS) phase**

   * Jitter optimizer
   * Reconfiguration-aware dependency checker
   * Minimizes changes to the incumbent data plane

Additional optimizations include:

* Pipeline scheduling inspired by microarchitecture pipelines
* Circular shifting for multi-period flows
* Space and time reduction techniques

---

## Metrics

The scheduler estimates:

* **End-to-End Delay**, composed of:

  * Signal delay
  * Propagation delay
  * Processing delay
  * Queuing / scheduling delay

* **End-to-End Jitter**, defined as the difference between maximum and minimum delay across flow iterations

---

## Experimental Results

### Evaluation Setup

* Comparison between:

  * ILP (solved with Gurobi)
  * Heuristic with jitter optimization
* Fractional factorial Design of Experiments (DoE)
* 3000 randomly generated TS flows
* Simulations executed on an OpenStack VM (4 vCores, 32 GB RAM)

### Key Findings

* The heuristic achieves high feasibility ratios close to ILP
* Significant reduction in execution time
* Acceptable optimality gap
* Effective support for heterogeneous wired and wireless interfaces

---

## Contributions

* Deterministic, reconfiguration-aware TAS scheduling
* Support for heterogeneous network interfaces
* Multi-period TS flow handling
* Pipeline-based latency minimization
* Practical applicability to provider transport networks

---

## Applications

* Industrial automation
* Autonomous vehicles and ITS
* Energy networks
* Healthcare systems
* Financial infrastructures

---

## Related Publication

> L. Velasco, G. Graziadei, Y. El Kaisi, J. Villares, O. Muñoz, J. Vidal, M. Ruiz
> *Provisioning of Time-Sensitive and Non-Time-Sensitive Flows: from Control to Data Plane*
> IFIP Networking 2024 – TENSOR Workshop
>
> L. Velasco, G. Graziadei, S. Barzegar and M. Ruiz, "Provisioning of Time-Sensitive and Non-Time-Sensitive Flows With Assured Performance," in IEEE Transactions on Network and Service Management, vol. 22, no. 2, pp. 1484-1499, April 2025, doi: 10.1109/TNSM.2025.3539697.

---

## Future Work

* Validation with discrete-event simulators (ns-3, OMNeT++)
* Guard band vs throughput trade-off analysis
* Dynamic programming approaches
* Support for dynamic TS flow requirements

---

## License

This work is provided for academic and research purposes. Please cite appropriately if used in publications.
