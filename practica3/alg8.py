from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Parámetros

N_ESTIMATORS=500
LR=1.1
BASE_ESTIMATOR=DecisionTreeClassifier(max_depth=12)

model = AdaBoostClassifier(n_estimators=N_ESTIMATORS, learning_rate=LR, base_estimator=BASE_ESTIMATOR, random_state=25)
