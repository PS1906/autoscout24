import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Daten laden
df = pd.read_csv('data/autoscout24.csv').dropna(subset=['price','mileage','hp','offerType'])

# Feature-Matrix aufbauen (inkl. Dummies für make und offerType)
X = pd.get_dummies(df[['mileage','hp','make','offerType']], drop_first=True)
y = df['price']

# Modell trainieren
model = LinearRegression().fit(X, y)

# Modell speichern
joblib.dump(model, 'pricing_model.pkl')
print("Modell gespeichert als pricing_model.pkl")
