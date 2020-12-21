from sklearn.svm import SVC

# Parámetros
C=65

model = SVC(C=C, class_weight='balanced', random_state=25)
