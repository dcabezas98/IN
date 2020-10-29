import numpy as np
import matplotlib.pyplot as plt

models=["Dummy","DecisionTree","GaussianNB","SupportVectorM","RandomForest","KNN","NeuralNetwork"]

tpr1=[1.0,0.77,0.8525,0.7425,0.805,0.845,0.8425]
fpr1=[1.0,0.1059,0.2,0.2988,0.1835,0.2541,0.2188]

tpr2=[1.0,0.7573,0.8404,0.6697,0.773,0.827,0.8225]
fpr2=[1.0,0.1124,0.1996,0.2616,0.188,0.2267,0.2054]

tpr3=[1.0,0.7725,0.865,0.74,0.805,0.8525,0.8075]
fpr3=[1.0,0.1059,0.1835,0.2965,0.1906,0.2682,0.2047]

tpr4=[1.0,0.77,0.885,0.695,0.8125,0.8225,0.855]
fpr4=[1.0,0.1012,0.2447,0.3129,0.2,0.2212,0.2071]

tpr5=[1.0,0.77,0.885,0.86,0.8175,0.82,0.8325]
fpr5=[1.0,0.1012,0.2447,0.1741,0.2024,0.1624,0.1647]

TPR=tpr5
FPR=fpr5

for i, (x,y) in enumerate(zip(FPR,TPR)):
    print(x,y)
    plt.plot(x,y, 'o', label=models[i])
    
plt.legend(loc='lower right')
plt.axis([0,1,0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
