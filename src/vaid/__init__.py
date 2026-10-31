"""Visibility-aware open-world IoT device identification.

Reference implementation of the framework described in
docs/proposal/VisibilityAware_IoT_Device_Identification.docx, built to run on
data/samples (UNSW-IoTraffic proof-of-concept slice).

Module map:
    config      constants the manuscript fixes by name
    naming      device-key normalisation (the WithingsSmartScale trap)
    ingest      flow and protocol-parameter readers, join-rate measurement
    handshake   ClientHello / DHCP parsing under the forbidden-field spec
    audit       §4.1 dataset audit and §4.3 fail-closed feature audit
    windows     observation construction and the three views
    views       train-only vocabularies and scalers, feature matrices
    visibility  profiles V0-V6 and view dropout
    perturb     Experiment 8 traffic-perturbation simulation
    splits      E1-E5 partitions
    models_*    encoders, gated fusion, fusion baselines, classical baselines
    openset     prototype and energy rejection with disjoint calibration classes
    metrics     macro-F1, OSCR, calibration, bootstrap and Wilcoxon
    experiments drivers for Experiments 1-5, 8, 9, 10
    cli         command line
"""

__version__ = "0.1.0"
