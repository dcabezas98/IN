import numpy as np

def na(train, test):

    del train['Descuento']
    del test['Descuento']
    train.dropna(inplace=True)

    return train, test
    
def preprocessing(train, test):    

    train_array = np.array(train)
    label=train_array[:,-1]
    train_array = train_array[:,:-1]

    test_array = np.array(test)

    return train_array, label, test_array
