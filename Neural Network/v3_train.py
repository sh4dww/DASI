from sklearn import preprocessing
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

X = []
Y = []

# read the training data
with open("../Train_trec06p.csv") as f:
    for line in f:
        curr = line.split(",")
        new_curr = [1]
        for item in curr[:len(curr) - 2]:
            new_curr.append(float(item))
        X.append(new_curr)
        Y.append([float(curr[-1])])
X = np.array(X)
X = preprocessing.scale(X) # feature scaling
y = np.array(Y)

print(len(y[0]))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5499, test_size=3435, random_state=5)

print(len(X_train[0]))

# Define el número de pasos de tiempo
timesteps = 10  

# Número de características de los datos de entrada
features = 57  

# Inicializar el modelo
model = Sequential()

model.add(Dense(57, input_dim=57, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Cargar los pesos guardados
model.load_weights('v3.weights.h5')  

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Definir el punto de control para guardar los pesos del modelo
checkpoint = ModelCheckpoint('modelo.keras', save_best_only=True)

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Guardar el modelo entrenado
model.save_weights('v3.weights.h5')

# Suponiendo que ya tienes tus datos de prueba y sus etiquetas
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print('Precisión del modelo:', test_accuracy)

"""# printing the output
print ("total = ", len(layer_2))
print ("correct = ", correct)
print ("accuracy = ", correct * 100.0 / len(layer_2))"""