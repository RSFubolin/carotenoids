# carotenoids
This project includes crucial test data and code.
# Description
This repository provides a complete multi-task BPINNs pipeline to predict leaf carotenoid content using spectral reflectance data. Key technical innovations:
Multi-task Bayesian Network: Four variational output heads for carotenoids, tree_high, chl, lma based on TensorFlow Probability DenseVariational layers.
Composite Custom Loss Function: Combine primary regression loss, auxiliary multi-task constraint loss, physical boundary loss for carotenoids, and KL divergence regularization for Bayesian weights.
GPU Acceleration: Automatic GPU memory growth configuration, TF graph compilation for fast training.
Hyperparameter Random Search: Automated trial-based hyperparameter tuning with configurable search space (network layers, dropout, activation, optimizer, batch size, epochs).
Uncertainty Quantification: Monte Carlo sampling prediction variance, 95% confidence interval coverage rate (CR) evaluation to validate Bayesian uncertainty reliability.
Standardized Output: Auto-save Excel reports including train/test predictions, performance metrics, optimal hyperparameter configuration, and uncertainty statistics.
Built-in Metrics: Calculate \(R^2\), RMSE, MAE, coverage rate of 95% confidence intervals for model evaluation.
# Prerequisites
pip install numpy pandas tensorflow tensorflow-probability matplotlib scikit-learn openpyxl.
# Hardware Requirements
GPU Recommended: NVIDIA GPU with CUDA support (automatically detected; CPU fallback supported)
Minimum RAM: 8GB (16GB+ recommended for large spectral datasets)
Storage: Local disk for input Excel data and output result folders
Python ≥ 3.8
TensorFlow ≥ 2.9
TensorFlow Probability ≥ 0.17
Scikit-learn ≥ 1.0
Openpyxl for Excel I/O
# Core Execution Flow
Auto detect GPU devices and enable memory growth to avoid out-of-memory errors.
Load and parse Excel data, split into TensorFlow train/test datasets.
Initialize random hyperparameter search (default 10 trials, adjustable n_trials).
For each trial:
Construct multi-task Bayesian PINN with sampled hyperparameters
Train with composite custom loss, gradient clipping, epoch-wise KL weight scheduling
Evaluate test performance, record model metrics
Save all hyperparameter trial records to Excel.
Retain best-performing model, compute Monte Carlo prediction uncertainty and confidence interval coverage rate.
Export detailed prediction results, performance comparison table, and optimal configuration to species-named output folders.
# Contact
e-mail: fubolin@glut.edu.cn,1020232077@glut.edu.cn
If you have questions about model structure, remote sensing data adaptation, hyperparameter tuning or uncertainty quantification methods, you can contact me for communication and technical support.
# Citation
If you use this code for academic research, please cite this repository.
