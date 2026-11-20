# Individual Project Proposal: A Compact CIFAR-10 Classifier for CPU Deployment

**Student:** Habtamu  **Student ID:** 70568  **Date:** 20 November 2026

## Problem

Image classifiers that reach good accuracy are usually too large and too slow for edge devices. This project builds a small image classifier that runs on an ordinary CPU, with no GPU, and still stays accurate. The goal is to take a large, accurate "teacher" network and compress its knowledge into a much smaller "student" network that meets strict size and speed limits.

## Dataset

CIFAR-10: 60,000 colour images of 32×32 pixels in 10 classes (50,000 for training, 10,000 for testing). I will hold out 5,000 training images for validation and use the 10,000 test images only for the final results. The dataset is public and downloads through `torchvision`.

## Success Metric

The primary metric is top-1 accuracy on the CIFAR-10 test set. The project succeeds if the final compressed model reaches **≥ 85% test accuracy** (minimum acceptable result: 80%) while meeting both deployment constraints below. I will also report how much accuracy the compression costs compared with the teacher.

## Deployment Constraints

| Constraint | Limit | How it will be measured |
|---|---|---|
| Inference latency | ≤ 50 ms per image | Batch size 1 on my CPU, median of 1,000 runs after a 100-run warm-up |
| Model size | ≤ 5 MB | Size of the saved model file on disk |

## Planned Optimization

1. **Teacher:** train a larger CNN (ResNet-style) to get a strong accuracy reference.
2. **Knowledge distillation:** train a small student CNN (about 0.3 million parameters) using the teacher's soft predictions (temperature T = 4) mixed with the normal label loss. I will compare it with the same student trained without distillation to show the benefit.
3. **Quantization:** apply post-training static int8 quantization to the student to reduce file size and CPU latency further.

If time allows, I will add light pruning as an extra experiment.

## Evaluation Plan

I will report one table with four models (teacher, student without distillation, student with distillation, student with distillation + int8), showing test accuracy, file size in MB and CPU latency in ms. All runs will use my Lab 6 pipeline: a single YAML config, a fixed seed, checkpoints, and MLflow tracking of parameters, metrics and artifacts, so every result is reproducible.

## Plan and Risks

- **Week 1:** CIFAR-10 data loading, teacher training, baseline measurements.
- **Week 2:** student baseline and distillation experiments.
- **Week 3:** quantization, latency and size measurements, results table.
- **Week 4:** analysis and final report.

*Main risk:* training the teacher on CPU is slow. Mitigation: use a modest teacher size, a limited number of epochs, and the profiling and batch-size findings from this lab to choose efficient settings. If the 85% target is out of reach in the time available, I will report the best achieved accuracy (at least 80%) and analyse the gap.