import streamlit as st

st.set_page_config(page_title="Accueil - AVC", page_icon="🧠", layout="centered")

st.title("🧠 Bienvenue sur l'outil de prédiction du risque d'AVC")

# ✅ Image locale
st.image("streamlit_app/AVC.png", use_container_width=True, caption="Illustration locale - Risque d'AVC")

# ✅ Description du projet
st.markdown("""
## 🧠 Pourquoi ce projet ?

Les **accidents vasculaires cérébraux (AVC)** sont une cause majeure de **mortalité et de handicap**.

Ce projet vise à fournir un outil d’aide à la décision simple pour **détecter les patients à risque**  
et permettre une **prise en charge précoce**.

## 🎯 Objectifs
- Identifier les profils à risque à partir de données médicales
- Offrir une interface intuitive pour les professionnels de santé

## 🔍 Explorer les fonctionnalités dans le menu latéral

> Utilisez le menu ⬅️ à gauche pour accéder à la **prédiction** et à la page **À propos**.
""")
