from sklearn.neural_network import MLPClassifier

# Parámetros
HIDDEN_LAYER_SIZES=(200,200)

model = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZES, random_state=25, max_iter=1000)
