from sklearn.ensemble import RandomForestClassifier

# Parámetros
N_ESTIMATORS=350
MAX_DEPTH=20

model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, n_jobs=4, random_state=25)
