import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

# Cargar el conjunto de datos
data = pd.read_csv('/Users/cristhianrecalde/Downloads/winequality-red.csv')

# Separar las características y la variable objetivo
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Binarizar la variable objetivo
#>= 7 alta calidad
#<7 baja calidad
y = np.where(y >= 7, 1, 0)

# Dividir el conjunto de datos en entrenamiento y prueba
#20% datos se usan para la prueba
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Construir el modelo de red neurona
model = Sequential()
model.add(Dense(9, activation='relu', input_shape=(11,)))
model.add(Dense(9, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(X_test, y_test)
print("Precisión:", accuracy)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).flatten()

# Crear la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Realizar predicciones para nuevos conjuntos de datos
new_data1 = np.array([[10.80, 1.16, 0.60, 8.86, 0.27, 46.86, 129.84, 1.0022, 3.96, 0.97, 13.55]])
new_data2 = np.array([[9.31, 1.17, 0.00, 5.31, 0.10, 7.56, 58.71, 0.9948, 3.24, 1.23, 11.1]])

new_data1_scaled = scaler.transform(new_data1)
new_data2_scaled = scaler.transform(new_data2)

prediction1 = model.predict(new_data1_scaled)
prediction2 = model.predict(new_data2_scaled)

print("Predicción para el conjunto de datos 1:"+ str(prediction1), "Alta calidad" if prediction1 > 0.5 else "Baja calidad ")
print("Predicción para el conjunto de datos 2:"+str(prediction2), "Alta calidad" if prediction2 > 0.5 else "Baja calidad ")