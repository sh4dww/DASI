import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Cargar el archivo CSV
data = pd.read_csv('../Train2.csv')

# Dividir los datos en características (X) y etiquetas (y)
X = data.iloc[:, :-1]  # Todas las columnas excepto la última
y = data.iloc[:, -1]   # Última columna

# Dividir los datos en 2500 filas para entrenamiento y 500 filas para pruebas
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, train_size=5499, test_size=3435, random_state=5)

print(y_test)
print(y_train)

np.save("train.npy", {
    "X_train": X_train,
    "y_train": y_train
})

np.save("test.npy", {
    "X_test": X_test,
    "y_test": y_test
})