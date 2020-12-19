
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold

DATA='ugrin2020-vehiculo-usado-multiclase/'
NOMBRE=DATA+'nombre.csv'
CIUDAD=DATA+'ciudad.csv'
COMBUSTIBLE=DATA+'combustible.csv'
TIPO_MARCHAS=DATA+'tipo_marchas.csv'

def encodeNorm(train, test):

    le = LabelEncoder()

    # Me quedo con la marca del coche
    train.Nombre=train.Nombre.apply(lambda x: x.split(' ')[0])
    test.Nombre=test.Nombre.apply(lambda x: x.split(' ')[0])

    # Codifico las marcas
    nombres=pd.read_csv(NOMBRE,header=0)
    nombres.Nombre=nombres.Nombre.apply(lambda x: x.split(' ')[0])
    le.fit(nombres.Nombre)
    train.Nombre = le.transform(train.Nombre)
    test.Nombre = le.transform(test.Nombre)

    # Codifico las ciudades
    ciudades = pd.read_csv(CIUDAD,header=0)
    le.fit(ciudades.Ciudad)
    train.Ciudad = le.transform(train.Ciudad)
    test.Ciudad = le.transform(test.Ciudad)

    # Codifico combustibles
    combustibles = pd.read_csv(COMBUSTIBLE, header=0)
    le.fit(combustibles.Combustible)
    train.Combustible = le.transform(train.Combustible)
    test.Combustible = le.transform(test.Combustible)

    # Codifico tipo marchas
    tipo_marchas = pd.read_csv(TIPO_MARCHAS, header=0)
    le.fit(tipo_marchas.Tipo_marchas)
    train.Tipo_marchas = le.transform(train.Tipo_marchas)
    test.Tipo_marchas = le.transform(test.Tipo_marchas)

    # Codifico manos
    le.fit(['First','Second','Third','Fourth & Above'])
    train.Mano = le.transform(train.Mano)
    test.Mano = le.transform(test.Mano)

    # Consumo a numérica
    train.Consumo=train.Consumo.apply(lambda x: float(x.split(' ')[0]))
    test.Consumo=test.Consumo.apply(lambda x: float(x.split(' ')[0]))
    
    # CC a numérica
    train.Motor_CC=train.Motor_CC.apply(lambda x: float(x.split(' ')[0]))
    test.Motor_CC=test.Motor_CC.apply(lambda x: float(x.split(' ')[0]))

    # Potencia a numérica
    train.Potencia=train.Potencia.apply(lambda x: float(x.split(' ')[0]))
    test.Potencia=test.Potencia.apply(lambda x: float(x.split(' ')[0]))

    # Estandarizamos los datos

    selector = VarianceThreshold()
    std = StandardScaler()

    return train, test
