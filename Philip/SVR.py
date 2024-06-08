import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# Lade die CSV-Datei (angenommen, die Datei heißt "data.csv")
data = pd.read_csv("initial_data.csv")

# Teile die Daten in Features (X) und Ziel (y) auf
X = data.drop(columns=["Engine speed", "Engine load", "Railpressure", "Air supply", "Crank angle", "Intake pressure", "Back pressure", "Intake temperature"]).values
y = data[["NOx", "PM 1", "CO2", "PM 2", "Pressure cylinder"]].values

# Teile die Daten in Trainings- und Testsets auf
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisiere die Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Trainiere ein Support Vector Regressor (SVR)-Modell
svr = SVR(kernel="linear", C=1.0, epsilon=0.1)
multi_output_svr = MultiOutputRegressor(svr)
multi_output_svr.fit(X_train_scaled, y_train)

# Mache Vorhersagen auf dem Testset
y_pred = multi_output_svr.predict(X_test_scaled)

# Zeige die SVR-Vorhersagen in einem Diagramm
plt.xlim(0,500)
plt.ylim(0,500)
plt.scatter(y_test[:, 0], y_pred[:, 0], color="red", label="NOx")
plt.scatter(y_test[:, 1], y_pred[:, 1], color="green", label="PM 1")
plt.scatter(y_test[:, 2], y_pred[:, 2], color="blue", label="CO2")
plt.scatter(y_test[:, 3], y_pred[:, 3], color="orange", label="PM 2")
plt.scatter(y_test[:, 4], y_pred[:, 4], color="purple", label="Pressure cylinder")
plt.xlabel("Tatsächlich")
plt.ylabel("Vorhergesagt")
plt.title("SVR-Vorhersagen")
plt.legend()
plt.show()


# Berechne die Abweichung (Residuen) für jedes Ausgabeattribut
residuen = y_test - y_pred

# Berechne den durchschnittlichen absoluten Fehler für jedes Ausgabeattribut
mae = np.mean(np.abs(residuen), axis=0)
for i in range(0,5):
    mse = mean_squared_error(y_test[:,i], y_pred[:,i])
    #print(str(i) + ". mse: " + str(mse))
    varianz = np.var(y_test[:,i])
    print("normalisiert \t" + str(mse/varianz))

# Gib die durchschnittlichen absoluten Fehler für jedes Ausgabeattribut aus
for i in range(y_test.shape[1]):
    print(f"Durchschnittlicher absoluter Fehler für Ausgabespalte {i+1}: {mae[i]}")

# Gib eine Erfolgsmeldung aus
print("SVR-Modell erfolgreich trainiert!")