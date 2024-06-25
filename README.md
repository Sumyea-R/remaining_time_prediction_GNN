## Remaining Time Prediction with Graph Neural Network
This repository focuses on predicting remaining time of a process using graph neural network from logistics dataset. We first preprocess and create features from the data and divide it to train and test set. 
Then we train the data using GNN, tune hyperparameters and evaluate the results following the implementation of the paper **"Remaining cycle time prediction with Graph Neural Networks for Predictive Process Monitoring"**.

## Installation
The project is implemented with jupyter notebook in both python and pytorch.

- pytorch: Follow the instructions in the PyTorch website https://pytorch.org/get-started/locally/
- torch geometric for GNN implementation: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
- Ax package for hyperparamaters tuning: pip3 install ax-platform
- PM4Py: pip install pm4py

## Reproduction

- Run main_rtm.ipnyb notebook to preprocess data, create feature set and produce train and test data set
- Run RCT_prediction_Gated_GNN.ipynb notebook to train models and evaluate the results.

