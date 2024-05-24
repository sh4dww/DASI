import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

train_data = np.load("train.npy", allow_pickle=True)
test_data = np.load("test.npy", allow_pickle=True)

X_train_loaded = train_data.item().get("X_train")
y_train_loaded = train_data.item().get("y_train")
X_test = test_data.item().get("X_test")
y_test = test_data.item().get("y_test")

#print(X_test)

X_train = np.copy(X_train_loaded)
y_train = np.copy(y_train_loaded)

data = pd.read_csv('../Train2.csv')

# Dividir los datos en características (X) y etiquetas (y)
X = data.iloc[:, :-1]  # Todas las columnas excepto la última
y = data.iloc[:, -1]   # Última columna

#print(X)

# Crear el clasificador de Bosques Aleatorios
classifier = RandomForestClassifier(n_estimators=30, random_state=5)

# Entrenar el clasificador
classifier.fit(X_train, y_train)

#print(y_test)

# Hacer predicciones en el conjunto de pruebas
y_prediction = classifier.predict(X_test)

# Calcular la precisión del modelo
accuracy = metrics.accuracy_score(y_test, y_prediction)
print("Precisión del modelo:", accuracy)

#print(y_prediction)

# Otras funciones disponibles
classifier.apply(X.values)
classifier.decision_path(X.values)
classifier.get_params(deep=True)
classifier.predict_proba(X.values)
classifier.score(X.values, y.values)

np.save("train.npy", {
    "X_train": X_train,
    "y_train": y_train
})

np.save("test.npy", {
    "X_test": X_test,
    "y_test": y_test
})

conf_matrix = confusion_matrix(y_test, (y_prediction > 0.5))

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                              display_labels=['Spam', 'No Spam'])

disp.plot()
plt.show()