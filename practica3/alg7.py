from sklearn.ensemble import GradientBoostingClassifier

# Parámetros

N_ESTIMATORS=500
LR=0.1

model = GradientBoostingClassifier(n_estimators=N_ESTIMATORS, learning_rate=LR, random_state=25)
