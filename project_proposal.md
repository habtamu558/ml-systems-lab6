# Project Proposal

**Title:** Lightweight Coffee Bean Roast-Level Classification: Comparing MLP, CNN and Attention Models Under Edge Latency and Size Budget

**Name:** Habtamu Teshome 
**Student ID:** 70568

## Problem
Coffee roast level (green, light, medium, dark) is usually checked by eye, so the result can change from person to person. I want to build a small image classifier that could run on a cheap device such as a Raspberry Pi or a phone. In this project I will compare three types of models (MLP, CNN and attention) and see which one gives the best accuracy while staying small and fast.

## Dataset
I plan to use the public Coffee Bean Dataset from Kaggle, which has 4 classes: Green, Light, Medium and Dark. I will check the exact number of images and the license on the dataset page. The plan is:

- Keep the official test set untouched.
- Split the training images into train and validation, stratified.
- Resize the images to 64×64.
- Use simple augmentation (flip, small rotation) on the training data only.
- Check for duplicate images, as I did in Lab 4.

## Success Metric
- Test accuracy and F1 score. I hope to reach about 90% or more, but I will report whatever I get.
- A confusion matrix, because I expect Light and Medium to be confused.
- Number of parameters, model size and latency for each model.

## Deployment Constraints
- Latency of 50 ms or less per image on CPU (batch size 1).
- Model size of 5 MB or less.

## Planned Optimization
I will build three models with a similar number of parameters, like in Lab 3:

1. **MLP:** a few Linear and ReLU layers.
2. **CNN:** two or three Conv and MaxPool blocks, then a Linear layer.
3. **Attention:** split the image into patches and use a small Transformer encoder.

Then I will apply **quantization (INT8)** to the models and compare the FP32 and INT8 versions on accuracy, size and latency. If quantization works well and I have time left, I may try pruning or distillation, but that is optional.

## Tools
- **PyTorch and torchvision** for models and training.
- **YAML config and a fixed seed** so the runs can be repeated (Lab 6).
- **MLflow** to record parameters and results.
- **torchinfo and thop** to count parameters and FLOPs.
- **torch.profiler** to find the slow parts of training.
- **scikit-learn and matplotlib** for metrics and plots.
- **Git** to save my code and report.

## How I Will Evaluate
- Train each model with Adam and the same settings.
- Measure latency as the median of 100 runs after warm-up, on the same computer.
- Check model size from the saved file.
- Run each model with a few different seeds to see if the results are stable.
- Make a table and two plots (accuracy vs size, accuracy vs latency), then check which models meet the 50 ms and 5 MB limits.

## Plan
- **Week 1:** prepare the data and check its quality.
- **Week 2:** train the three models.
- **Week 3:** quantization and latency tests.
- **Week 4:** plots, report and final commit.

## Possible Problems
- The dataset is small, so the attention model may overfit. I will use augmentation and early stopping.
- Quantization may reduce accuracy. If it does, I will report that and try quantization-aware training.
- The photos in the dataset may not match real lighting conditions. I will mention this as a limitation.
