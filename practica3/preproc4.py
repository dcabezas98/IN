import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold


DATA='ugrin2020-vehiculo-usado-multiclase/'
NOMBRE=DATA+'nombre.csv'
CIUDAD=DATA+'ciudad.csv'
COMBUSTIBLE=DATA+'combustible.csv'
TIPO_MARCHAS=DATA+'tipo_marchas.csv'

VARIANCETHRESHOLD=0

# Tratar los valores perdidos
def na(train, test):

    del train['Descuento']
    del test['Descuento']
    train.dropna(inplace=True)

    return train, test

# Codificar los datos 
def encode(train, test):

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

    return train, test

# Binarización de Categóricas
def binarize(train, test):

    le = LabelEncoder()
    oneHot= OneHotEncoder(sparse=False, drop='first')

    # Codifico las marcas
    nombres=pd.read_csv(NOMBRE,header=0)
    nombres.Nombre=nombres.Nombre.apply(lambda x: x.split(' ')[0])
    marcas_enc=le.fit_transform(nombres.Nombre)
    oneHot.fit(marcas_enc.reshape(-1,1))
    marca_bin_train=oneHot.transform(train[:,0].reshape(-1, 1))
    marca_bin_test=oneHot.transform(test[:,0].reshape(-1, 1))

    # Codifico las ciudades
    ciudades = pd.read_csv(CIUDAD,header=0)
    ciudades_enc=le.fit_transform(ciudades.Ciudad)
    oneHot.fit(ciudades_enc.reshape(-1,1))
    ciudad_bin_train=oneHot.transform(train[:,1].reshape(-1, 1))
    ciudad_bin_test=oneHot.transform(test[:,1].reshape(-1, 1))

    # Codifico los combustibles
    combustibles = pd.read_csv(COMBUSTIBLE, header=0)
    combustibles_enc=le.fit_transform(combustibles.Combustible)
    oneHot.fit(combustibles_enc.reshape(-1,1))
    combustible_bin_train=oneHot.transform(train[:,4].reshape(-1, 1))
    combustible_bin_test=oneHot.transform(test[:,4].reshape(-1, 1))

    # Codifico tipo marchas
    tipo_marchas = pd.read_csv(TIPO_MARCHAS, header=0)
    marchas_enc=le.fit_transform(tipo_marchas.Tipo_marchas)
    oneHot.fit(marchas_enc.reshape(-1,1))
    marchas_bin_train=oneHot.transform(train[:,5].reshape(-1, 1))
    marchas_bin_test=oneHot.transform(test[:,5].reshape(-1, 1))

    train=np.hstack((marca_bin_train,ciudad_bin_train,train[:,2:4],combustible_bin_train,marchas_bin_train,train[:,6:]))
    test=np.hstack((marca_bin_test,ciudad_bin_test,test[:,2:4],combustible_bin_test,marchas_bin_test,test[:,6:]))

    return train, test
    
# Normalización
def scale(train, test):

    # Estandarizamos los datos
    selector = VarianceThreshold(VARIANCETHRESHOLD) # No podemos estandarizar datos con varianza nula
    std = StandardScaler()

    selector.fit(train)
    train=selector.transform(train)
    test=selector.transform(test)

    std.fit(train)
    train=std.transform(train)
    test=std.transform(test)
    
    return train, test

# Split train label
def split(train, test):

    train_array = np.array(train)
    label=train_array[:,-1]
    train_array = train_array[:,:-1]

    test_array = np.array(test)

    return train_array, label, test_array

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

# Preprocesado
def preprocessing(train, test):

    train, test = na(train, test)
    
    train, test = encode(train, test)

    train, label, test = split(train, test)

    train, test = binarize(train, test)
          
    shuffle_in_unison(train, label)

    train, test = scale(train, test)
    
    return train, label, test
