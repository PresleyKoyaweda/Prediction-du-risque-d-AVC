import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="📊 Monitoring des Prédictions", page_icon="📈")
st.title("📊 Monitoring des prédictions en temps réel")

log_file = "logs/predictions.csv"

if not os.path.exists(log_file):
    st.warning("📭 Aucune donnée de prédiction trouvée pour le moment.")
    st.stop()

# Chargement des prédictions
df = pd.read_csv(log_file)

# Vérifie que les colonnes sont correctes
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
else:
    st.error("⛔ La colonne 'timestamp' est absente du fichier CSV.")
    st.stop()

# Statistiques de base
st.subheader("📌 Statistiques générales")
st.write(df[["predictions", "probability"]].describe())

# Répartition des prédictions
st.subheader("📊 Répartition des prédictions")
df["predictions_label"] = df["predictions"].map({0: "Pas d'AVC", 1: "AVC"})
st.bar_chart(df["predictions_label"].value_counts(normalize=True))

# Évolution temporelle
st.subheader("🕒 Évolution temporelle de la probabilité")
if len(df) < 10:
    st.warning("⏳ Pas encore assez de prédictions pour afficher une moyenne glissante (minimum 10).")
else:
    st.line_chart(df.set_index("timestamp")["probability"].rolling(window=10).mean())

# Données brutes
with st.expander("📄 Voir les données brutes"):
    st.dataframe(df.tail(20))
