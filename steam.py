import os
import kagglehub  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

path = kagglehub.dataset_download("nikdavis/steam-store-games")
csv_path = os.path.join(path, "steam.csv")

df = pd.read_csv(csv_path)

print(df.columns)

df = df[['name', 'release_date', 'price', 'developer', 'positive_ratings', 'negative_ratings']] 
df = df.dropna()  

df['metacritic_score'] = (df['positive_ratings'] - df['negative_ratings']) / (df['positive_ratings'] + df['negative_ratings']) * 100
df['metacritic_score'] = df['metacritic_score'].fillna(0).astype(int)  

df['label'] = df['metacritic_score'].apply(lambda x: 1 if x > 80 else 0)

df['release_year'] = pd.to_datetime(df['release_date']).dt.year

df['developer'] = LabelEncoder().fit_transform(df['developer'])

df = df.drop(columns=['name', 'release_date', 'positive_ratings', 'negative_ratings'])

print(df.head())

X = df.drop('label', axis=1)  
y = df['label']  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

modelos = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'Naive Bayes': GaussianNB()
}

resultados = {}

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    resultados[nome] = {
        'Acurácia': acc,
        'Matriz de Confusão': cm,
        'Relatório': report
    }

    print(f"\n===== {nome} =====")
    print("Acurácia:", acc)
    print("Matriz de Confusão:\n", cm)
    print("Relatório:\n", classification_report(y_test, y_pred))

acuracias = {nome: round(r['Acurácia'], 3) for nome, r in resultados.items()}
acuracias_df = pd.DataFrame.from_dict(acuracias, orient='index', columns=['Acurácia'])

plt.figure(figsize=(8, 5))
sns.barplot(x=acuracias_df.index, y='Acurácia', data=acuracias_df)
plt.ylim(0, 1)
plt.title('Comparação de Acurácias dos Modelos')
plt.ylabel('Acurácia')
plt.xlabel('Algoritmo')
plt.tight_layout()
plt.savefig("comparacao_acuracia.png")
plt.show()
