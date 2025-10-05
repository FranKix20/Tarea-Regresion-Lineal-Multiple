"""
Nombre: Franco Quintuman
Materia: Fundamentos de Data Science
Seccion: 302
Profesor: HECTOR EDUARDO CIFUENTES MELLA

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


'''
Queremos predecir el precio (en UF) de departamentos en Santiago en base a su superficie,
numero de habitaciones y distancia al metro.
'''

datos = {
    'Superficie_m2': [50, 70, 65, 90, 45],
    'Num_Habitaciones': [1, 2, 2, 3, 1],
    'Distancia_Metro_km': [0.5, 1.2, 0.8, 0.2, 2.0],
    'Precio_UF': [2500, 3800, 3500, 5200, 2100]
}

df = pd.DataFrame(datos)

X = df[['Superficie_m2', 'Num_Habitaciones', 'Distancia_Metro_km']]
y = df['Precio_UF']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Resultados de la Evaluación:")
print(f"RMSE: {rmse:.2f} UF")
print(f"MAE: {mae:.2f} UF (Error absoluto promedio)")
print(f"R-Cuadrado (R^2): {r2:.2f} (El modelo explica el {r2:.0%} de la variación en el precio)")


print("\nPrediccion de un nuevo departamento:")
# supongamos un depto de 60 m2, 2 habitaciones y a 0.7 km del metro
nuevo_depto = pd.DataFrame([[60, 2, 0.7]], columns=['Superficie_m2', 'Num_Habitaciones', 'Distancia_Metro_km'])
pred = model.predict(nuevo_depto)
print(f"Precio estimado: {pred[0]:.0f} UF")

print("\nConclusion:")
print("El modelo entrega una estimación del precio de departamentos segun sus caracteristicas. "
      "Sin embargo, debido al pequeño tamaño de la muestra, los resultados pueden no ser completamente confiables, "
      "aunque sirven para ilustrar cómo usar una regresión lineal multiple en este contexto.")