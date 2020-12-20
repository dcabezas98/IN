from sklearn.neighbors import KNeighborsClassifier

# Parámetros
N_NEIGHBORS=2
WEIGHTS='distance'
METRIC='manhattan'

model = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights=WEIGHTS, metric=METRIC, n_jobs=4)
