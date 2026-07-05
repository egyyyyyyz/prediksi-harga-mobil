import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os, warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Prediksi Harga Mobil Bekas", page_icon="🚗", layout="wide")

st.markdown("""
<style>
.main-title{text-align:center;font-size:2.2rem;font-weight:700;color:#1F4E79;margin-bottom:.2rem}
.sub-title{text-align:center;font-size:1rem;color:#888;margin-bottom:2rem}
.result-box{background:linear-gradient(135deg,#1a7a4a,#27ae60);border-radius:16px;padding:2rem;text-align:center;color:white;margin-top:1rem}
.result-price{font-size:2.8rem;font-weight:800;color:white}
.result-range{margin-top:.5rem;font-size:.95rem;color:white;opacity:.9}
.result-algo{margin-top:.5rem;font-size:.85rem;color:white;opacity:.75}
.info-card{background-color:#1e2a3a;border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:.8rem;border-left:4px solid #27ae60}
.info-label{font-size:.78rem;color:#aab8c8;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.2rem}
.info-value{font-size:1.1rem;font-weight:700;color:#ffffff}
.model-card{background-color:#1e2a3a;border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:.8rem;text-align:center}
.model-card.best{border:2px solid #f0ad00}
.model-name{font-size:1rem;font-weight:700;color:#ffffff;margin-bottom:.8rem}
.model-r2{font-size:2rem;font-weight:800;color:#27ae60}
.model-metric-label{font-size:.75rem;color:#aab8c8;margin-top:.6rem}
.model-metric-val{font-size:1rem;font-weight:600;color:#ffffff}
.car-badge{display:inline-block;background:#1F4E79;color:white;border-radius:8px;padding:4px 12px;font-size:.85rem;margin-top:.3rem}
</style>
""", unsafe_allow_html=True)

KURS = 190

def format_rupiah(angka):
    return "Rp {:,.0f}".format(angka).replace(",", ".")

@st.cache_resource
def load_and_train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'CAR_DETAILS_FROM_CAR_DEKHO.csv')
    df_raw   = pd.read_csv(csv_path)
    df       = df_raw.copy()

    # Ekstrak merek dan buat mapping merek → daftar tipe mobil
    df_raw['brand'] = df_raw['name'].apply(lambda x: x.split()[0])
    brands = sorted(df_raw['brand'].unique().tolist())
    brand_to_models = {}
    for brand in brands:
        models_list = sorted(df_raw[df_raw['brand'] == brand]['name'].unique().tolist())
        brand_to_models[brand] = models_list

    # Preprocessing
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    Q1  = df['selling_price'].quantile(0.25)
    Q3  = df['selling_price'].quantile(0.75)
    IQR = Q3 - Q1
    df  = df[(df['selling_price'] >= Q1-1.5*IQR) & (df['selling_price'] <= Q3+1.5*IQR)]
    df['car_age'] = 2025 - df['year']

    fuel_cats   = sorted(df['fuel'].unique().tolist())
    seller_cats = sorted(df['seller_type'].unique().tolist())
    trans_cats  = sorted(df['transmission'].unique().tolist())
    owner_cats  = sorted(df['owner'].unique().tolist())
    year_min    = int(df['year'].min())
    year_max    = int(df['year'].max())
    km_max      = int(df['km_driven'].max())

    le_fuel   = LabelEncoder().fit(df['fuel'])
    le_seller = LabelEncoder().fit(df['seller_type'])
    le_trans  = LabelEncoder().fit(df['transmission'])
    le_owner  = LabelEncoder().fit(df['owner'])

    df['fuel']          = le_fuel.transform(df['fuel'])
    df['seller_type']   = le_seller.transform(df['seller_type'])
    df['transmission']  = le_trans.transform(df['transmission'])
    df['owner']         = le_owner.transform(df['owner'])
    df['selling_price'] = np.log1p(df['selling_price'])

    X = df[['km_driven','fuel','seller_type','transmission','owner','car_age']]
    y = df['selling_price']

    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    ml_models = {
        'KNN Regression':    KNeighborsRegressor(n_neighbors=9),
        'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    trained = {}
    for name, model in ml_models.items():
        model.fit(X_train, y_train)
        pred   = np.expm1(model.predict(X_test))
        actual = np.expm1(y_test)
        results[name] = {
            'MAE':  mean_absolute_error(actual, pred) * KURS,
            'RMSE': np.sqrt(mean_squared_error(actual, pred)) * KURS,
            'R2':   r2_score(actual, pred)
        }
        trained[name] = model

    encoders = {'fuel': le_fuel, 'seller': le_seller, 'trans': le_trans, 'owner': le_owner}
    meta = {
        'fuel_cats': fuel_cats, 'seller_cats': seller_cats,
        'trans_cats': trans_cats, 'owner_cats': owner_cats,
        'year_min': year_min, 'year_max': year_max, 'km_max': km_max,
        'brands': brands, 'brand_to_models': brand_to_models
    }
    return trained, results, scaler, encoders, meta

# ── Header ─────────────────────────────────────────────────
st.markdown('<div class="main-title">🚗 Prediksi Harga Mobil Bekas</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Berbasis Machine Learning — Dataset Car Dekho | Harga dalam Rupiah</div>', unsafe_allow_html=True)
st.divider()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Info Aplikasi")
    st.markdown("---")
    st.markdown("**Dataset:** Car Dekho")
    st.markdown("**Jumlah data:** 4.340 baris")
    st.markdown("**Merek tersedia:** 29 merek")
    st.markdown("**Tipe mobil:** 1.491 tipe")
    st.markdown("**Algoritma:** KNN, Random Forest, Gradient Boosting")
    st.markdown("---")
    st.caption(f"Kurs: 1 INR ≈ Rp {KURS:,}")
    st.caption("Universitas Dian Nuswantoro")

# ── Load model ──────────────────────────────────────────────
try:
    with st.spinner("⏳ Memuat data dan melatih model..."):
        trained, results, scaler, encoders, meta = load_and_train()
    st.success("✅ Model berhasil dilatih dan siap digunakan!")
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

# ── Tabs ───────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi Harga", "📊 Performa Model", "ℹ️ Cara Pakai"])

# ════════ TAB 1: PREDIKSI ════════
with tab1:
    st.subheader("Masukkan Spesifikasi Mobil")

    # ── Pilih Merek & Tipe Mobil ──
    st.markdown("#### 🚘 Pilih Merek & Tipe Mobil")
    col_brand, col_type = st.columns(2)
    with col_brand:
        brand_choice = st.selectbox("🏭 Merek Mobil", meta['brands'])
    with col_type:
        tipe_list    = meta['brand_to_models'][brand_choice]
        tipe_choice  = st.selectbox(f"🚗 Tipe Mobil ({len(tipe_list)} tersedia)", tipe_list)

    st.markdown("---")
    st.markdown("#### ⚙️ Detail Kendaraan")

    col1, col2 = st.columns(2)
    with col1:
        tahun = st.slider("📅 Tahun Produksi", min_value=meta['year_min'], max_value=meta['year_max'], value=2018)
        km    = st.number_input("🛣️ Jarak Tempuh (km)", min_value=0, max_value=meta['km_max'], value=45000, step=1000)
        fuel  = st.selectbox("⛽ Jenis Bahan Bakar", meta['fuel_cats'])
    with col2:
        seller       = st.selectbox("🏪 Tipe Penjual", meta['seller_cats'])
        transmission = st.selectbox("⚙️ Jenis Transmisi", meta['trans_cats'])
        owner        = st.selectbox("👤 Status Kepemilikan", meta['owner_cats'])
        model_choice = st.selectbox("🤖 Pilih Algoritma", list(trained.keys()))

    st.markdown("---")

    if st.button("🔮 PREDIKSI HARGA SEKARANG", use_container_width=True, type="primary"):
        car_age    = 2025 - tahun
        fuel_enc   = encoders['fuel'].transform([fuel])[0]
        seller_enc = encoders['seller'].transform([seller])[0]
        trans_enc  = encoders['trans'].transform([transmission])[0]
        owner_enc  = encoders['owner'].transform([owner])[0]

        X_input   = np.array([[km, fuel_enc, seller_enc, trans_enc, owner_enc, car_age]])
        X_scaled  = scaler.transform(X_input)
        harga_inr = np.expm1(trained[model_choice].predict(X_scaled))[0]
        harga_rp  = harga_inr * KURS
        min_rp    = harga_rp * 0.85
        max_rp    = harga_rp * 1.15

        st.markdown(f"""
        <div class="result-box">
            <div style="font-size:.9rem;color:white;opacity:.85;margin-bottom:.3rem">🚗 {tipe_choice}</div>
            <div style="font-size:1rem;color:white;margin-bottom:.5rem">✅ Estimasi Harga Jual Mobil Bekas</div>
            <div class="result-price">{format_rupiah(harga_rp)}</div>
            <div class="result-range">Rentang wajar: {format_rupiah(min_rp)} — {format_rupiah(max_rp)}</div>
            <div class="result-algo">Algoritma: {model_choice} &nbsp;|&nbsp; Usia: {car_age} tahun &nbsp;|&nbsp; {km:,} km</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**📋 Ringkasan Input:**")

        r1, r2, r3 = st.columns(3)
        r4, r5, r6 = st.columns(3)
        r7, r8, _  = st.columns(3)

        r1.markdown(f'<div class="info-card"><div class="info-label">🏭 Merek</div><div class="info-value">{brand_choice}</div></div>', unsafe_allow_html=True)
        r2.markdown(f'<div class="info-card"><div class="info-label">🚗 Tipe Mobil</div><div class="info-value">{tipe_choice}</div></div>', unsafe_allow_html=True)
        r3.markdown(f'<div class="info-card"><div class="info-label">📅 Tahun</div><div class="info-value">{tahun}</div></div>', unsafe_allow_html=True)
        r4.markdown(f'<div class="info-card"><div class="info-label">🛣️ Jarak Tempuh</div><div class="info-value">{km:,} km</div></div>', unsafe_allow_html=True)
        r5.markdown(f'<div class="info-card"><div class="info-label">⛽ Bahan Bakar</div><div class="info-value">{fuel}</div></div>', unsafe_allow_html=True)
        r6.markdown(f'<div class="info-card"><div class="info-label">⚙️ Transmisi</div><div class="info-value">{transmission}</div></div>', unsafe_allow_html=True)
        r7.markdown(f'<div class="info-card"><div class="info-label">🏪 Tipe Penjual</div><div class="info-value">{seller}</div></div>', unsafe_allow_html=True)
        r8.markdown(f'<div class="info-card"><div class="info-label">👤 Kepemilikan</div><div class="info-value">{owner}</div></div>', unsafe_allow_html=True)

# ════════ TAB 2: PERFORMA ════════
with tab2:
    st.subheader("📊 Perbandingan Performa Model")
    st.caption("Evaluasi menggunakan 20% data uji | Harga dalam Rupiah")

    best_model = max(results, key=lambda x: results[x]['R2'])
    icons = ["🔵","🟢","🟡"]
    cols  = st.columns(3)

    for i, (name, metrics) in enumerate(results.items()):
        is_best = name == best_model
        badge   = " ⭐ TERBAIK" if is_best else ""
        border  = "best" if is_best else ""
        with cols[i]:
            st.markdown(f"""
            <div class="model-card {border}">
                <div class="model-name">{icons[i]} {name}{badge}</div>
                <div class="model-r2">{metrics['R2']*100:.2f}%</div>
                <div class="model-metric-label">R² Score (Akurasi)</div>
                <hr style="border-color:#2e3d50;margin:.8rem 0">
                <div class="model-metric-label">MAE</div>
                <div class="model-metric-val">{format_rupiah(metrics['MAE'])}</div>
                <div class="model-metric-label" style="margin-top:.4rem">RMSE</div>
                <div class="model-metric-val">{format_rupiah(metrics['RMSE'])}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**Tabel Perbandingan Lengkap:**")
    df_result = pd.DataFrame({
        'Model':    list(results.keys()),
        'R² Score': [f"{v['R2']*100:.2f}%" for v in results.values()],
        'MAE':      [format_rupiah(v['MAE'])  for v in results.values()],
        'RMSE':     [format_rupiah(v['RMSE']) for v in results.values()],
    })
    st.dataframe(df_result, use_container_width=True, hide_index=True)
    st.info(f"⭐ Model terbaik: **{best_model}** dengan R² = {results[best_model]['R2']*100:.2f}%")

    st.divider()

    # ── Daftar Merek & Jumlah Tipe ──
    st.markdown("**🚘 Daftar Merek Mobil dalam Dataset:**")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'CAR_DETAILS_FROM_CAR_DEKHO.csv')
    df_info  = pd.read_csv(csv_path)
    df_info['brand'] = df_info['name'].apply(lambda x: x.split()[0])
    brand_count = df_info.groupby('brand').agg(
        Jumlah_Data=('name','count'),
        Jumlah_Tipe=('name','nunique')
    ).reset_index().rename(columns={'brand':'Merek'})
    brand_count = brand_count.sort_values('Jumlah_Data', ascending=False).reset_index(drop=True)
    st.dataframe(brand_count, use_container_width=True, hide_index=True)

# ════════ TAB 3: CARA PAKAI ════════
with tab3:
    st.subheader("ℹ️ Cara Menggunakan Aplikasi")
    st.markdown("""
    ### 🚀 Langkah-langkah:
    1. Buka tab **"Prediksi Harga"**
    2. **Pilih Merek Mobil** (29 merek tersedia: Maruti, Toyota, Honda, dll)
    3. **Pilih Tipe Mobil** sesuai merek yang dipilih (1.491 tipe tersedia)
    4. Isi detail kendaraan (tahun, km, bahan bakar, transmisi, penjual, kepemilikan)
    5. Pilih algoritma yang ingin digunakan
    6. Klik **"PREDIKSI HARGA SEKARANG"**
    7. Hasil estimasi harga dalam **Rupiah** muncul beserta rentang harga wajar

    ---
    ### 🏭 Merek Mobil yang Tersedia:
    `Ambassador` `Audi` `BMW` `Chevrolet` `Daewoo` `Datsun` `Fiat` `Force` `Ford`
    `Honda` `Hyundai` `Isuzu` `Jaguar` `Jeep` `Kia` `Land Rover` `MG` `Mahindra`
    `Maruti` `Mercedes-Benz` `Mitsubishi` `Nissan` `Renault` `Skoda` `Tata`
    `Toyota` `Volkswagen` `Volvo`

    ---
    ### 🤖 Algoritma yang Tersedia:
    | Algoritma | Cara Kerja |
    |-----------|------------|
    | **KNN Regression** | Mencari K mobil paling mirip lalu merata-ratakan harganya |
    | **Random Forest** | Menggabungkan banyak pohon keputusan untuk prediksi akurat |
    | **Gradient Boosting** | Membangun model bertahap, memperbaiki error sebelumnya |

    ---
    ### 💱 Konversi Harga:
    - Dataset asli dalam **Rupee India (INR)**
    - Dikonversi otomatis ke **Rupiah Indonesia (IDR)**
    - Kurs: **1 INR ≈ Rp 190**

    ---
    ### 📌 Catatan:
    - Rentang ±15% adalah estimasi harga wajar
    - Semakin tinggi R² Score, semakin akurat modelnya
    - Dataset: Car Dekho (4.340 data, 29 merek, 1.491 tipe)
    """)

st.divider()
st.caption("🎓 Penelitian Machine Learning | Prediksi Harga Mobil Bekas | Dataset: Car Dekho | Universitas Dian Nuswantoro")