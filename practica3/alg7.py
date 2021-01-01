from sklearn.ensemble import GradientBoostingClassifier

# Parámetros

N_ESTIMATORS=550
LR=0.15
SUBSAMPLE=0.9
MAX_DEPTH=2

model = GradientBoostingClassifier(n_estimators=N_ESTIMATORS, learning_rate=LR, subsample=SUBSAMPLE, max_depth=MAX_DEPTH,random_state=25)
