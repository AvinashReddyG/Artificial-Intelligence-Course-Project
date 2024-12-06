## Revisiting the NN vs. GBDT Debate: Insights from the Tabular Data Analysis
This project aims to provide insights into the performance comparison between the NN and GBDT models. 
The implementation compares the performance of **Neural Networks** and **Gradient Boosting Decision trees** on tabular datasets.

## Algorithms and Datasets
The project focuses on testing 12 algorithms across 76 datasets from OpenML-CC18 suite and the OpenML Benchmarking Suite. 
The algorithms include:
- Two GBDTmodels: XGBoost and LightGBM.
- Six neural networks: MLP, ResNet, TabNet, NODE, SAINT and VIME.
- Four baseline algorithms: K-Nearest Neighbors (KNN), Logistic Regression, Decision Tree and Random Forest.

Datasets with above 50,000 tuples were dropped.

## Implementation
We start by importing necessary libraries such as pandas, openml for Baseline models, Torch for the neural network, and **XGBoost** for the boosted trees.
Before training, we preprocess the data. We load datasets from **OpenML** using their task IDs and handle missing values. 
We also scale the features and split the data into training and testing sets to ensure the models are evaluated on unseen data.
We define the algorithms for Baseline models, GBDTs and NNs. 
We utilize a simple feedforward network with **fully connected layers** and **ReLU activations** for neural network architecture. The network's depth and width can be tuned.
We evaluate them using **accuracy** as our metric.
After training the datasets, we compare their **performance**. 
By plotting the accuracy for both neural networks and boosted trees across different datasets, we can identify patterns and draw conclusions on when one model outperforms the other.
We also extract the metafeatures of the datasets to understand the features affecting the performance.

## Steps to reproduce the Code:

- Open the file "Complete Code of all algotithms".
- Upload the requirements.txt (this txt file is for "Complete Code of all algotithms.ipynb" file only) to the Google Colab or the directory where you have the python file.
- Run the first cell shown which will install the libraries required to run the code:
**!pip install --no-deps -r requirements.txt**
- Proceed with running the rest of the code. 


## Reference
```bibtex
@inproceedings{mcelfresh2023neural,
  title={When Do Neural Nets Outperform Boosted Trees on Tabular Data?}, 
  author={McElfresh, Duncan and Khandagale, Sujay and Valverde, Jonathan and Ramakrishnan, Ganesh and Prasad, Vishak and Goldblum, Micah and White, Colin}, 
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}, 
} 
```
