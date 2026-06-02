import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import xgboost as xgb
from scipy.sparse import hstack

# ---------------------------------------------------------------
# KONFIGURASI DASAR HALAMAN
# ---------------------------------------------------------------
st.set_page_config(page_title="SPK Prediksi Sukses Game Steam", page_icon="🎮", layout="wide")

# ---------------------------------------------------------------
# FUNGSI LOAD SEMUA MODEL & ENCODER
# ---------------------------------------------------------------
@st.cache_resource
@st
def load_assets():
    """Memuat semua aset (model, encoder, SHAP background) hanya sekali."""
    # 1️⃣ Model XGBoost (.pkl)
    with open('xgboost_model_full.pkl', 'rb') as f:
        model_obj = pickle.load(f)
    booster = model_obj.get_booster()

    # 2️⃣ Background data (untuk SHAP)
    background_df = pd.read_csv("background_data.csv")

    # 3️⃣ Encoder dan metadata
    with open('encoder_platform.pkl', 'rb') as f: encoder_platform = pickle.load(f)
    with open('encoder_cat.pkl', 'rb') as f: encoder_cat = pickle.load(f)
    with open('encoder_gen.pkl', 'rb') as f: encoder_gen = pickle.load(f)
    with open('encoder_tags.pkl', 'rb') as f: encoder_tags = pickle.load(f)
    with open('semua_nama_fitur.pkl', 'rb') as f: all_feature_names = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f: le = pickle.load(f)

    # 4️⃣ Daftar opsi dari encoder teks
    opsi_categories = list(encoder_cat.get_feature_names_out())
    opsi_genres = list(encoder_gen.get_feature_names_out())
    opsi_tags = list(encoder_tags.get_feature_names_out())

    # 5️⃣ Buat TreeExplainer SHAP (lebih cepat dari KernelExplainer)
    explainer = shap.TreeExplainer(booster, feature_perturbation="tree_path_dependent")

    return (
        model_obj, booster, explainer, background_df,
        encoder_platform, encoder_cat, encoder_gen, encoder_tags,
        all_feature_names, le, opsi_categories, opsi_genres, opsi_tags
    )

# Muat aset
(
    model_obj, booster, explainer, background_data,
    encoder_platform, encoder_cat, encoder_gen, encoder_tags,
    all_feature_names, le, opsi_categories, opsi_genres, opsi_tags
) = load_assets()

# ---------------------------------------------------------------
# ANTARMUKA STREAMLIT
# ---------------------------------------------------------------
st.title("🎮 Sistem Pendukung Keputusan: Prediksi Keberhasilan Game Steam")
st.sidebar.header("Masukkan Atribut Game")

# ---- Input Numerik ----
price = st.sidebar.slider("💲 Harga Game (USD)", 0.0, 70.0, 19.99, 0.5)
required_age = st.sidebar.selectbox("🔞 Rating Usia", [0, 3, 7, 12, 16, 18], index=0)
achievements = st.sidebar.number_input("🏆 Jumlah Achievements", min_value=0, value=50, step=10)
metacritic_score = st.sidebar.slider("📊 Metacritic Score", 0, 100, 75)
user_score = st.sidebar.slider("⭐ User Score", 0, 100, 80)
recommendations = st.sidebar.number_input("👍 Jumlah Rekomendasi", min_value=0, value=200, step=10)
dlc_count = st.sidebar.number_input("🧩 Jumlah DLC", min_value=0, value=1, step=1)

# ---- Input Platform ----
st.sidebar.subheader("🖥️ Platform yang Didukung")
platform_win = st.sidebar.checkbox("Windows", value=True)
platform_mac = st.sidebar.checkbox("Mac", value=False)
platform_linux = st.sidebar.checkbox("Linux", value=False)

# ---- Input Kategorikal ----
default_cat = ["singleplayer"] if "singleplayer" in opsi_categories else []
default_gen = ["action"] if "action" in opsi_genres else []
default_tag = ["indie"] if "indie" in opsi_tags else []

selected_categories = st.sidebar.multiselect("Kategori", opsi_categories, default=default_cat)
selected_genres = st.sidebar.multiselect("Genre", opsi_genres, default=default_gen)
selected_tags = st.sidebar.multiselect("Tags", opsi_tags, default=default_tag)

# Tombol prediksi
predict_button = st.sidebar.button("🚀 Lakukan Prediksi", type="primary")

# ---------------------------------------------------------------
# LOGIKA PREDIKSI
# ---------------------------------------------------------------
if predict_button:
    with st.spinner("🤖 Menganalisis atribut game Anda... Harap tunggu beberapa detik."):
        # --- 1️⃣ Siapkan input ---
        input_numerik_df = pd.DataFrame([[
            required_age, price, dlc_count, metacritic_score,
            user_score, achievements, recommendations
        ]], columns=[
            'Required age', 'Price', 'DiscountDLC count', 'Metacritic score',
            'User score', 'Achievements', 'Recommendations'
        ])

        input_platform_df = pd.DataFrame([[platform_win, platform_mac, platform_linux]],
                                         columns=['Windows', 'Mac', 'Linux'])

        input_cat = [" ".join(selected_categories)]
        input_gen = [" ".join(selected_genres)]
        input_tags = [" ".join(selected_tags)]

        # --- 2️⃣ Encoding ---
        numerik_transformed = input_numerik_df.astype(float).values
        platform_transformed = encoder_platform.transform(input_platform_df)
        cat_transformed = encoder_cat.transform(input_cat)
        gen_transformed = encoder_gen.transform(input_gen)
        tags_transformed = encoder_tags.transform(input_tags)

        # --- 3️⃣ Gabungkan fitur ---
        input_sparse = hstack([
            numerik_transformed, platform_transformed,
            cat_transformed, gen_transformed, tags_transformed
        ]).tocsr()

        input_df_for_plot = pd.DataFrame(input_sparse.toarray(), columns=all_feature_names)

        # --- 4️⃣ Prediksi model ---
        dmatrix_input = xgb.DMatrix(input_sparse)
        prob_sangat_sukses = booster.predict(dmatrix_input)[0]
        pred_label_index = 1 if prob_sangat_sukses >= 0.5 else 0
        pred_label = le.classes_[pred_label_index]

        # --- 5️⃣ Analisis SHAP ---
        shap_values = explainer.shap_values(input_df_for_plot)
        if isinstance(shap_values, list):
            shap_vals_to_plot = shap_values[pred_label_index][0]
            base_value = explainer.expected_value[pred_label_index]
        else:
            shap_vals_to_plot = shap_values[0]
            base_value = explainer.expected_value

        # -----------------------------------------------------------
        # OUTPUT HASIL
        # -----------------------------------------------------------
        st.markdown("## 🎯 **Hasil Analisis AI Game Anda**")
        col1, col2 = st.columns(2)
        col1.metric("Prediksi", pred_label)
        col2.metric("Probabilitas 'Sangat Sukses'", f"{prob_sangat_sukses*100:.2f}%")

        st.divider()
        st.markdown("### 🔍 Faktor-Faktor yang Mempengaruhi Prediksi")

        try:
            shap_plot = shap.force_plot(
                base_value,
                shap_vals_to_plot,
                input_df_for_plot.iloc[0],
                matplotlib=False
            )
            shap_html = f"<head>{shap.getjs()}</head><body>{shap_plot.html()}</body>"
            st.components.v1.html(shap_html, height=300)
        except Exception as e:
            st.error(f"⚠️ Gagal menampilkan SHAP plot: {e}")

        st.divider()
        st.markdown("### 📋 **Detail Atribut Game**")
        st.dataframe(input_df_for_plot.T.loc[input_df_for_plot.T[0] > 0].rename(columns={0: "Nilai"}))

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.markdown(
    """
    ---
    🧠 **Catatan Teknis:**
    - Model: XGBoost (binary logistic)
    - Interpretasi: SHAP TreeExplainer
    - Dibangun untuk mendukung pengambilan keputusan data game Steam
    - 👨‍💻 *Dinsos Web by: Teknik Informatika UNIMA*
    """
)
