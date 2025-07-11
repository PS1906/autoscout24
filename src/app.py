# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# 1) Modell laden (als .pkl oder .joblib exportiert)
@st.cache_data(allow_output_mutation=True)
def load_model(path='src/models/pricing_model.pkl'):
    return joblib.load(path)

model = load_model()

# 2) Daten laden oder manuell eingeben
@st.cache_data
def load_data(path='src/data/autoscout24.csv'):
    df = pd.read_csv(path).dropna(subset=['price','mileage','hp','offerType'])
    return df

df = load_data()

st.title("Pricing-Modell Dashboard")

# 3) Sidebar-Filter
st.sidebar.header("Filter")
make_sel      = st.sidebar.multiselect("Marke",    df['make'].unique())
offer_sel     = st.sidebar.multiselect("OfferType", ['Used','Demonstration','Pre-registered'])
year_sel      = st.sidebar.slider("Jahr", int(df.year.min()), int(df.year.max()), (2018,2021))

# 4) Gefilterte Daten
mask = df['year'].between(*year_sel)
if make_sel:  mask &= df['make'].isin(make_sel)
if offer_sel: mask &= df['offerType'].isin(offer_sel)
df_filt = df[mask]

st.subheader(f"Gefilterte Daten: {len(df_filt)} Fahrzeuge")
st.dataframe(df_filt[['make','year','offerType','mileage','hp','price']].head(50))

# 5) Vorhersage für gefilterte Daten
X = df_filt[['mileage','hp']]
preds = model.predict(X)
df_filt = df_filt.assign(pred_price=preds)

# 6) Visualisierung
fig, ax = plt.subplots()
ax.scatter(df_filt['mileage'], df_filt['price'], label='Ist-Preis', alpha=0.4)
ax.scatter(df_filt['mileage'], df_filt['pred_price'], label='Pred-Preis', alpha=0.6, marker='x')
ax.set_xlabel('Kilometerstand')
ax.set_ylabel('Preis')
ax.legend()
st.pyplot(fig)

# 7) Zusammenfassung
st.markdown(f"""
**Modell-Performance**  
- Datensätze: {len(df)}  
- R² (Test): *hier einfügen*  
- Features: mileage, hp, make, offerType  
""")
