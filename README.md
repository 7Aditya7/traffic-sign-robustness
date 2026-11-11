# Everest Intelligence — Traffic Sign Classification Analysis

## Project overview

This project explores how a pretrained Vision Transformer classifies traffic signs and how its predictions change when images are modified. We use the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset, which has **43 traffic-sign classes**, and the pretrained model [`kelvinandreas/vit-traffic-sign-GTSRB`](https://huggingface.co/kelvinandreas/vit-traffic-sign-GTSRB).

Our focus is on **model errors, prediction confidence, and robustness to image changes**. We use notebooks, plots, and an interactive dashboard to identify patterns that may be missed when looking only at overall accuracy. Our analysis describes observed model behavior; it does not establish the model's internal reasons for making a prediction.

## Main analyses

### P1 — Misclassification patterns

Explores which traffic-sign classes the model confuses, how these errors change under image corruptions such as Gaussian noise and motion blur, and which individual predictions improve or worsen.

### P2 — Prediction confidence

Investigates confidence distributions and how the model's prediction confidence changes between the original test images and corrupted versions.

### P3 — Gaussian blur and class-specific behavior

Tests **eight Gaussian blur strengths** on the full **12,630-image GTSRB test set**. The analysis focuses on **Class 26 (Traffic Signals)**, which was frequently confused with **Class 18 (General Caution)**.

At blur radius 1, **overall test accuracy decreases from 95.38% to 88.67%**, while **Class 26 accuracy increases from 62.78% to 77.22%**. The notebook also examines changes in individual predictions, image characteristics, and representative examples. This demonstrates that overall accuracy can hide class-specific effects.

### P4 — Interactive model exploration

Provides shared inference outputs and an interactive dashboard to inspect classification performance, errors, and prediction changes across image conditions.


