import joblib
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import seaborn as sns

# Load model & data
model = joblib.load("hasil_prediksi.pkl")  # Ganti dengan model kamu
data = pd.read_csv("hasil_fitur.csv")  # Dataset asli (untuk visualisasi)

st.set_page_config(page_title="Prediksi Diskon", layout="wide")
# st.title("📉 Aplikasi Prediksi Diskon Berdasarkan Transaksi")🎯 

# Sidebar navigasi
menu = st.sidebar.selectbox("Pilih Halaman", [
    "Prediksi Diskon", "Visualisasi Fitur Penting", "Statistik Diskon"
])

# ========================== 1. Halaman Prediksi ================================
if menu == "Prediksi Diskon":
    st.header("Prediksi Diskon Berdasarkan Input Transaksi")

    if "predict_clicked" not in st.session_state:
        st.session_state.predict_clicked = False

    wib = pytz.timezone("Asia/Jakarta")
    now = datetime.now(wib)
    day = now.day
    month = now.month
    year = now.year
    hour = now.hour
    minute = now.minute

    col1, col2 = st.columns(2)
    with col1:
        quantity = st.number_input("Quantity", min_value=1, value=5)
        price = st.number_input("Price", min_value=0.0, value=50.0)

    st.markdown(f"**Tanggal saat ini**: {day}-{month}-{year}, pukul {hour}:{minute:02d} WIB")


    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Metode Pembayaran")
        payment_method = st.selectbox("Pilih metode", ["Cash", "Credit", "Debit", "PayPal"])
        payment_features = {
            "PaymentCash": 1 if payment_method == "Cash" else 0,
            "PaymentCredit": 1 if payment_method == "Credit" else 0,
            "PaymentDebit": 1 if payment_method == "Debit" else 0,
            "PaymentPayPal": 1 if payment_method == "PayPal" else 0,
        }
    with col2:
        st.subheader("Kategori Produk")
        product_category = st.selectbox("Pilih kategori", ["Books", "Clothing", "Electronics", "Home Decor"])
        product_features = {
            "ProductCategory_Books": 1 if product_category == "Books" else 0,
            "ProductCategory_Clothing": 1 if product_category == "Clothing" else 0,
            "ProductCategory_Electronics": 1 if product_category == "Electronics" else 0,
            "ProductCategory_HomeDecor": 1 if product_category == "Home Decor" else 0,
        }

    if st.button("🔍 Prediksi Diskon"):
        st.session_state.predict_clicked = True

    if st.session_state.predict_clicked:
        total_amount = quantity * price
     
        input_data = pd.DataFrame([{
            "Quantity": quantity,
            "Price": price,
            "TotalAmount": total_amount,
            "Day": day,
            "Month": month,
            "Year": year,
            "Hour": hour,
            **payment_features,
            **product_features
        }])

        prediction = model.predict(input_data)[0]
        st.success(f"🎉 Diskon yang diprediksi: {prediction:.2f}%")

        #totalAmpunt
        st.metric(label="💰 Total Amount sebelum prediksi", value=f"$ {total_amount:,.2f}")

        discounted_amount = total_amount * (1 - prediction / 100)
        st.success(f"💸 Total Amount setelah diskon: $ {discounted_amount:,.2f}")



# ======================= 2. Visualisasi Fitur Penting ========================
elif menu == "Visualisasi Fitur Penting":
    st.header("🔎 Fitur yang Paling Mempengaruhi Diskon")
    importances = model.feature_importances_
    feature_names = model.feature_names_in_  # Gunakan jika tersedia

    sorted_idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[sorted_idx], y=feature_names[sorted_idx])
    plt.title("Feature Importance")
    plt.xlabel("Pentingnya Fitur")
    plt.ylabel("Fitur")
    st.pyplot(plt)

# =========================== 3. Statistik Diskon ===========================
elif menu == "Statistik Diskon":
    st.header("📊 Statistik dan Distribusi Diskon")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Diskon")
        fig, ax = plt.subplots()
        sns.histplot(data["DiscountApplied(%)"], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Boxplot Diskon per Kategori Produk")
        fig2, ax2 = plt.subplots()
        df_melt = data.copy()
        df_melt["Category"] = df_melt[
            ["ProductCategory_Books", "ProductCategory_Clothing", "ProductCategory_Electronics", "ProductCategory_HomeDecor"]
        ].idxmax(axis=1).str.replace("ProductCategory_", "")
        sns.boxplot(data=df_melt, x="Category", y="DiscountApplied(%)", ax=ax2)
        st.pyplot(fig2)

