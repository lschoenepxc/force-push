import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import csv
import re

def read_csv_file(file_path):
    # Initialisiere eine leere Liste, um die bereinigten Daten zu speichern
    cleaned_data = []

    # Lese die CSV-Datei
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')  # Annahme: Semikolon als Trennzeichen
        for row in reader:
            # Verbinde die Zeilenelemente mit einem Komma, um das Trennzeichen zu vereinheitlichen
            unified_row = ','.join(row)
            # Ersetze mehrere aufeinander folgende Kommas durch ein einzelnes Komma
            unified_row = re.sub(r',+', ',', unified_row)
            # Teile die vereinheitlichte Zeile nach dem Komma auf
            split_row = unified_row.split(',')
            # Entferne die ersten zwei Parameter
            cleaned_row = split_row[1:]
            # Füge die bereinigte Zeile der Liste hinzu
            cleaned_data.append(cleaned_row)

    return cleaned_data

# Beispiel: Daten aus "input.csv" einlesen


def getPLSPrediction(inputData):
    data = pd.read_csv("initial_data.csv")
    cleaned_data = read_csv_file("querys_ForcePush.csv")
    df_queried = pd.DataFrame(cleaned_data[1:], columns=cleaned_data[0])
    def add_data(data, queried_data):
        # add queried data (without cost) to initial data
        queried_data = queried_data[queried_data['costs'] == 1]
        data = pd.concat([data, queried_data.iloc[:, :13]], axis=0)

        # data = data.append(queried_data, ignore_index=True)
        return data

    data = add_data(data, df_queried)
    data = data.reset_index().astype('Float32')
    data = data.drop(columns=['PM 2'])


    # Teile die Daten in Features (X) und Ziel (y) auf
    X = data[["Engine speed", "Engine load", "Railpressure", "Air supply", "Crank angle", "Intake pressure", "Back pressure", "Intake temperature"]].values
    y = data[["NOx", "PM 1", "CO2", "Pressure cylinder"]].values

    # PLS
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import mean_squared_error

    pls = PLSRegression(n_components=8)
    pls.fit(X, y)
    y_pred = pls.predict(inputData)
    return y_pred