import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv
import re


class PLS_Regression_Model:
    data = None
    pls2 = PLSRegression(n_components=8)

    def __init__(self):
        return

    def get_data(self, initial_data_path, new_data_path):
        # Initialisiere eine leere Liste, um die bereinigten Daten zu speichern
        cleaned_data = []
        # Lese die CSV-Datei
        with open(new_data_path, 'r') as csvfile:
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

            df_queried = pd.DataFrame(cleaned_data[1:], columns=cleaned_data[0])
            df_queried_2 = df_queried[df_queried['costs'] == str(1)]
            data = pd.concat([data, df_queried_2.iloc[:, :13]], axis=0)
            self.data = data.reset_index().astype('Float32')

    def train(self):
        self.pls2.fit(X_train, y_train)
        x = self.data[['Engine speed', 'Engine load', 'Railpressure', 'Air supply', 'Crank angle', 'Intake pressure', 'Back pressure', 'Intake temperature']]
        y = self.data[['NOx', 'PM 1', 'CO2', 'Pressure cylinder']]

        print(len(x))
        print(len(y))

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)


    def predict(self, X_test):
        return(self.pls2.predict(X_test))
    
myModel = PLS_Regression_Model
myModel.get_data("initial_data.csv", "querys_ForcePush.csv")
myModel.train()
submission_data = pd.read_csv("submission.csv")
y_pred = myModel.predict(submission_data)

print(y_pred)

