from sklearn.ensemble import GradientBoostingClassifier

# Parámetros

N_ESTIMATORS=500
#LR=0.175
#SUBSAMPLE=0.7
#MAX_DEPTH=6

#model = GradientBoostingClassifier(n_estimators=N_ESTIMATORS, learning_rate=LR, subsample=SUBSAMPLE, max_depth=MAX_DEPTH,random_state=25)
model = GradientBoostingClassifier(n_estimators=N_ESTIMATORS,random_state=25)
