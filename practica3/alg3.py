from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes=100, early_stopping=True, random_state=25, max_iter=1000)
