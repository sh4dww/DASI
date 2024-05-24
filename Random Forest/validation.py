import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Cargar el archivo CSV
data = pd.read_csv('../Train.csv')

# Dividir los datos en características (X) y etiquetas (y)
X = data.iloc[:, :-1]  # Todas las columnas excepto la última
y = data.iloc[:, -1]   # Última columna

n_estimators = 30

# Crear una instancia del modelo de clasificación
#modelo = LogisticRegression()
modelo = RandomForestClassifier(n_estimators, random_state=5)

# Crear una instancia de KFold con k=5
kf = KFold(n_splits=5)

# Realizar la validación cruzada
scores = cross_val_score(modelo, X, y, cv=kf)

print(f"Estimators: {n_estimators}")
# Imprimir los resultados de cada fold
for fold, score in enumerate(scores):
    print(f"Fold {fold+1}: {score}")

total_sum = sum(scores)
mean = total_sum / len(scores)
print(f"Mean: {mean}")