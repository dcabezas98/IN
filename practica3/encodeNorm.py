
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA='ugrin2020-vehiculo-usado-multiclase/'
NOMBRE=DATA+'nombre.csv'

def encodeNorm(train, test):

    le = LabelEncoder()

    # Me quedo con la marca del coche
    train.Nombre=train.Nombre.apply(lambda x: str(x).split(' ')[0])
    test.Nombre=test.Nombre.apply(lambda x: str(x).split(' ')[0])

    nombres=pd.read_csv(NOMBRE,header=0)
    nombres.Nombre=nombres.Nombre.apply(lambda x: str(x).split(' ')[0])
    le.fit(nombres.Nombre)
    # Codifico las marcas
    #train.Nombre = le.transform(train.Nombre)
    #test.Nombre = le.transform(test.Nombre)

    # CC a numérica
    train.Motor_CC=train.Motor_CC.apply(lambda x: float(x.split(' ')[0]))
    test.Motor_CC=test.Motor_CC.apply(lambda x: float(x.split(' ')[0]))

    return train, test
