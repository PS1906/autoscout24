# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Used Car Dashboard", layout="wide")

# --- Data loading ---
@st.cache_data
def load_data(path: str = 'data/autoscout24.csv') -> pd.DataFrame:
    return pd.read_csv(path)

df = load_data()

# --- Correlation matrices for 2011 and 2021 ---
features = ['price', 'hp', 'mileage']
corr_2011 = df[df['year'] == 2011][features].corr()
corr_2021 = df[df['year'] == 2021][features].corr()

# --- Top-5 automakers and their average prices ---
top5 = df['make'].value_counts().head(5).index.tolist()
df_top5 = df[df['make'].isin(top5)].copy()
avg_price_year = (
    df_top5
    .groupby(['year', 'make'])['price']
    .mean()
    .reset_index()
)
# Overall average price per automaker
avg_price_overall = (
    df_top5
    .groupby('make')['price']
    .mean()
    .round(2)
    .reset_index()
    .sort_values(by='price', ascending=False)
)

# --- App layout ---
st.title("Used Car Insights Dashboard")
st.markdown(
    "Analyse der Korrelationen und Durchschnittspreise der Top-5-Automarken."
)

# Display two heatmaps side by side
col1, col2 = st.columns(2)
with col1:
    st.subheader("Korrelationsmatrix 2011")
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(corr_2011, interpolation='nearest', cmap='Reds')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    for i in range(len(features)):
        for j in range(len(features)):
            ax.text(j, i, f"{corr_2011.iloc[i, j]:.2f}",
                    ha='center', va='center',
                    color='white' if abs(corr_2011.iloc[i, j]) > 0.5 else 'black')
    plt.tight_layout()
    st.pyplot(fig)
with col2:
    st.subheader("Korrelationsmatrix 2021")
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(corr_2021, interpolation='nearest', cmap='Reds')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    for i in range(len(features)):
        for j in range(len(features)):
            ax.text(j, i, f"{corr_2021.iloc[i, j]:.2f}",
                    ha='center', va='center',
                    color='white' if abs(corr_2021.iloc[i, j]) > 0.5 else 'black')
    plt.tight_layout()
    st.pyplot(fig)

# --- Line plot with multiselect and overall average table ---
st.subheader("Durchschnittspreise pro Jahr und Gesamtübersicht")
# Layout: two columns for table and chart
table_col, chart_col = st.columns([1, 2])
# Multiselect for automakers
table_col.markdown("**Top-5 Automarken Gesamt-Durchschnittspreis**")
table_col.dataframe(avg_price_overall.set_index('make'))

selected_makes = chart_col.multiselect(
    "Automarke(n) auswählen", options=top5, default=top5
)
filtered = avg_price_year[avg_price_year['make'].isin(selected_makes)] if selected_makes else pd.DataFrame()

if not filtered.empty:
    fig2 = px.line(
        filtered,
        x='year', y='price', color='make', markers=True,
        labels={
            'year': 'Verkaufsjahr',
            'price': 'Durchschnittspreis (€)',
            'make': 'Automarke'
        },
        hover_data={'price': ':.2f'}
    )
    chart_col.plotly_chart(fig2, use_container_width=True)
else:
    chart_col.info("Bitte mindestens eine Automarke auswählen, um den Plot anzuzeigen.")

# cd C:\Users\PC\Desktop\Projekte\Autoscout-Projekt\src
# streamlit run app.py