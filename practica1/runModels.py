SEED=185

import numpy as np
import os

from sklearn.model_selection import KFold
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

def runModels(x, y, name, models, model_names):
    outfilename='results/'+name+'.csv'
    TMPfile='TMP.csv'
    
    table = np.empty((len(models),10))
    
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
    table[:len(models),4]=(table[:len(models),0]+table[:len(models),1])/(table[:len(models),0]+table[:len(models),1]+table[:len(models),2]+table[:len(models),3])
    # TPR
    table[:len(models),5]=table[:len(models),0]/(table[:len(models),0]+table[:len(models),3])
    # FPR
    table[:len(models),6]=table[:len(models),2]/(table[:len(models),1]+table[:len(models),2])
    # AUC
    table[:len(models),7]=(1+table[:len(models),5]-table[:len(models),6])/2
    # F1-score
    table[:len(models),8]=2*table[:len(models),0]/(2*table[:len(models),0]+table[:len(models),2]+table[:len(models),3])
    # G-measure
    table[:len(models),9]=table[:len(models),0]/np.sqrt((table[:len(models),0]+table[:len(models),2])*(table[:len(models),0]+table[:len(models),3]))

    np.savetxt(TMPfile, table, delimiter=',', fmt=['%1.1f']*4+['%1.4f']*6)

    string=',"TP","TN","FP","FN","Acc","TPR","FPR","AUC","F1-score","G-measure"\n'
    with open(TMPfile) as tmp:
        i=0
        for line in tmp:
            string+=model_names[i]+','+line.replace('.0,',',')
            i+=1
    os.remove(TMPfile)
    with open(outfilename,"w+") as outf:
        outf.write(string)
