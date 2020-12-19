from sklearn.ensemble import RandomForestClassifier

# Parámetros
N_ESTIMATORS=100
MAX_DEPTH=None

model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, n_jobs=4)
