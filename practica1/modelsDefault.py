# Ejecuta los modelos con parámetros por defecto sobre unos datos que
# admite como parámetro. Calcula diferentes scores y los escribe en un
# csv
SEED=185

import numpy as np
import os

from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

# Para calcular la matriz de confusión usando validación cruzada sumamos las matrices obtenidas en las distintas particiones
# https://stats.stackexchange.com/questions/147175/how-is-the-confusion-matrix-reported-from-k-fold-cross-validation
# https://stackoverflow.com/questions/40057049/using-confusion-matrix-as-scoring-metric-in-cross-validation-in-scikit-learn
def KFoldConfusionMatrix(model, data, target):
    conf_matrix_list_of_arrays = []
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]
        model.fit(X_train, y_train)
        conf_matrix = confusion_matrix(y_test, model.predict(X_test))
        conf_matrix_list_of_arrays.append(conf_matrix)
    return np.sum(conf_matrix_list_of_arrays, axis=0)

def defaultModelsRun(x, y, name):
    outfilename='results/proc_'+name+'.csv'
    TMPfile='TMP.csv'
    
    table = np.empty((9,10))
    
    # Models:
    dummy=DummyClassifier(strategy='constant',constant=1)
    dt=tree.DecisionTreeClassifier(random_state=SEED)
    gnb=GaussianNB()
    svc=SVC(random_state=SEED)
    rf=RandomForestClassifier(n_jobs=4, random_state=SEED)
    knn=KNeighborsClassifier() # K=5 por defecto
    rn=MLPClassifier(max_iter=500,random_state=SEED) # Max_iter=500 porque recibí warning de convergencia
    models = [dummy, dt, gnb, svc, rf, knn, rn]

    for i,m in enumerate(models):
        conf_mat = KFoldConfusionMatrix(m,x,y)
        """ La matriz de confusión aparece como
          Pred:    0  1
        (Benigno) Real=0: TN FP
        (Maligno) Real=1: FN TP
        """
        # Quiero ponerla: TP, TN, FP, FN
        table[i,0:4]=[conf_mat[1,1],conf_mat[0,0],conf_mat[0,1],conf_mat[1,0]]

    # Acc
    table[:7,4]=(table[:7,0]+table[:7,1])/(table[:7,0]+table[:7,1]+table[:7,2]+table[:7,3])
    # TPR
    table[:7,5]=table[:7,0]/(table[:7,0]+table[:7,3])
    # FPR
    table[:7,6]=table[:7,2]/(table[:7,1]+table[:7,2])
    # AUC
    table[:7,7]=(1+table[:7,5]-table[:7,6])/2
    # F1-score
    table[:7,8]=2*table[:7,0]/(2*table[:7,0]+table[:7,2]+table[:7,3])
    # G-measure
    table[:7,9]=table[:7,0]/np.sqrt((table[:7,0]+table[:7,2])*(table[:7,0]+table[:7,3]))

    # Máximo
    table[7]=np.amax(table[1:7],axis=0)
    # Media
    table[8]=np.mean(table[1:7],axis=0)

    np.savetxt(TMPfile, table, delimiter=',', fmt=['%1.1f']*4+['%1.4f']*6)

    string=',"TP","TN","FP","FN","Acc","TPR","FPR","AUC","F1-score","G-measure"\n'
    rownames= ["Dummy","DecisionTree","GaussianNB","SupportVectorM","RandomForest","KNN","NeuralNetwork","Máximo","Media"]
    with open(TMPfile) as tmp:
        i=0
        for line in tmp:
            string+=rownames[i]+','+line.replace('.0,',',')
            i+=1
    os.remove(TMPfile)
    with open(outfilename,"w+") as outf:
        outf.write(string)
