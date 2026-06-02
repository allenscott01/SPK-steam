import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from scipy.sparse import hstack

st.set_page_config(
    page_title="SPK Prediksi Keberhasilan Game Steam",
    layout="wide"
)

@st.cache_resource
def load_assets():
    """Memuat semua aset model dan encoder hanya sekali."""
    try:
        with open('xgboost_model_full.pkl', 'rb') as f:
            model_obj = pickle.load(f)
        booster = model_obj.get_booster()

        with open('encoder_platform.pkl', 'rb') as f: encoder_platform = pickle.load(f)
        with open('encoder_cat.pkl', 'rb') as f: encoder_cat = pickle.load(f)
        with open('encoder_gen.pkl', 'rb') as f: encoder_gen = pickle.load(f)
        with open('encoder_tags.pkl', 'rb') as f: encoder_tags = pickle.load(f)
        
        with open('semua_nama_fitur.pkl', 'rb') as f: all_feature_names = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f: le = pickle.load(f)

        opsi_categories = list(encoder_cat.get_feature_names_out())
        opsi_genres = list(encoder_gen.get_feature_names_out())
        opsi_tags = list(encoder_tags.get_feature_names_out())
        
        fitur_numerik_dasbor = ['Required age', 'Price', 'DiscountDLC count', 'Achievements']

        return (
            model_obj, booster, 
            encoder_platform, encoder_cat, encoder_gen, encoder_tags,
            all_feature_names, le, opsi_categories, opsi_genres, opsi_tags,
            fitur_numerik_dasbor
        )
        
    except FileNotFoundError as e:
        st.error(f" **Error:** File aset tidak ditemukan! Pastikan file model Anda (.pkl, .csv) sudah benar. File hilang: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f" **Error saat memuat aset:** {e}")
        st.stop()
        
(
    model_obj, booster, 
    encoder_platform, encoder_cat, encoder_gen, encoder_tags,
    all_feature_names, le, opsi_categories, opsi_genres, opsi_tags,
    fitur_numerik_dasbor 
) = load_assets()

st.title(" Sistem Pendukung Keputusan Prediksi Keberhasilan Game Steam")
st.sidebar.header("Masukkan Atribut Game (Tahap Development)")

price = st.sidebar.slider("Harga Game (USD)", 0.0, 70.0, 19.99, 0.5)
required_age = st.sidebar.selectbox(" Rating Usia", [0, 3, 7, 12, 16, 18], index=0)
achievements = st.sidebar.number_input(" Jumlah Achievements", min_value=0, value=50, step=10)
dlc_count = st.sidebar.number_input(" Jumlah DLC", min_value=0, value=1, step=1)

st.sidebar.subheader(" Platform yang Didukung")
platform_win = st.sidebar.checkbox("Windows", value=True)
platform_mac = st.sidebar.checkbox("Mac", value=False)
platform_linux = st.sidebar.checkbox("Linux", value=False)

# Cek default value dengan aman
default_cat = ["singleplayer"] if "singleplayer" in opsi_categories else []
default_gen = ["action"] if "action" in opsi_genres else []
default_tag = ["indie"] if "indie" in opsi_tags else []
if not default_cat and opsi_categories: default_cat = [opsi_categories[0]]
if not default_gen and opsi_genres: default_gen = [opsi_genres[0]]
if not default_tag and opsi_tags: default_tag = [opsi_tags[0]]

selected_categories = st.sidebar.multiselect("Kategori", opsi_categories, default=default_cat)
selected_genres = st.sidebar.multiselect("Genre", opsi_genres, default=default_gen)
selected_tags = st.sidebar.multiselect("Tags", opsi_tags, default=default_tag)
predict_button = st.sidebar.button(" Lakukan Prediksi", type="primary")

if predict_button:
    with st.spinner(" Menganalisis atribut game Anda..."):

        try:
            input_numerik_df = pd.DataFrame([[
                required_age, price, dlc_count, achievements
            ]], columns=fitur_numerik_dasbor) 
        except Exception as e:
            st.error(f"Error saat membuat DataFrame numerik. Pastikan {fitur_numerik_dasbor} ada. Error: {e}")
            st.stop()

        input_platform_df = pd.DataFrame([[platform_win, platform_mac, platform_linux]],
                                         columns=['Windows', 'Mac', 'Linux'])

        input_cat = [" ".join(selected_categories)]
        input_gen = [" ".join(selected_genres)]
        input_tags = [" ".join(selected_tags)]

        try:
            numerik_transformed = input_numerik_df.astype(float).values
            platform_transformed = encoder_platform.transform(input_platform_df)
            cat_transformed = encoder_cat.transform(input_cat)
            gen_transformed = encoder_gen.transform(input_gen)
            tags_transformed = encoder_tags.transform(input_tags)

            input_sparse = hstack([
                numerik_transformed, platform_transformed,
                cat_transformed, gen_transformed, tags_transformed
            ]).tocsr()
        except Exception as e:
            st.error(f" Error saat transformasi data: {e}")
            st.stop()
            
        if input_sparse.shape[1] != len(all_feature_names):
            st.error(f"FATAL: Mismatch jumlah fitur! Input ({input_sparse.shape[1]}) vs Model ({len(all_feature_names)}). Harap perbarui file aset (pkl/csv) dari Colab.")
            st.stop()

        input_df_for_plot = pd.DataFrame(input_sparse.toarray(), columns=all_feature_names)
        dmatrix_input = xgb.DMatrix(input_sparse)
        
        prob_sangat_sukses = booster.predict(dmatrix_input)[0]
        pred_label_index = 1 if prob_sangat_sukses >= 0.5 else 0
        pred_label = le.classes_[pred_label_index]
        
    st.markdown("## Hasil Analisis Model XGboost Game Anda")
    col1, col2 = st.columns(2)
    col1.metric("Prediksi", pred_label)
    col2.metric("Probabilitas 'Tingkat Keberhasilan'", f"{prob_sangat_sukses*100:.2f}%")

    st.divider()
    with st.expander("Lihat Detail Atribut yang Dianalisis"):
        active_features = input_df_for_plot.T.loc[input_df_for_plot.iloc[0] > 0]
        st.dataframe(active_features.rename(columns={0: "Nilai"}))

# --- Footer ---
st.markdown(
    """
    ---
    - Model : XGBoost 
    - Framework : Streamlit
    - Dibangun untuk mendukung pengambilan keputusan tahap development game Steam
    """
)