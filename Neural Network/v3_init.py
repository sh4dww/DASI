from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Inicializar el modelo
model = Sequential()

# Añadir la primera capa con 57 entradas
model.add(Dense(57, input_dim=57, activation='relu'))

# Añadir la segunda capa con 4 neuronas
model.add(Dense(4, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Resumen del modelo
model.summary()

model.save_weights('v3.weights.h5')