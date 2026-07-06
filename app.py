"""
Probability of Default (PD) Prediction App
============================================
Aplikasi Streamlit untuk memprediksi probabilitas gagal bayar (default)
menggunakan model XGBoost yang sudah dilatih sebelumnya.

Cara pakai:
1. Letakkan file model (.pkl) dan metadata (.json) di folder yang sama
   dengan app.py ini, lalu sesuaikan MODEL_PATH & METADATA_PATH di bawah.
2. Jalankan: streamlit run app.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ============================================================
# KONFIGURASI
# ============================================================
MODEL_PATH = "xgb_pd_model.pkl"                         
METADATA_PATH = "xgb_pd_metadata.json"                  
PREPROCESSING_BUNDLE_PATH = "preprocessing_bundle.pkl"  
CLAUDE_MODEL = "claude-sonnet-5"                 

# Kolom mentah yang WAJIB ada di CSV yang diupload user
RAW_REQUIRED_COLUMNS = [
    "loan_id", "grade", "home_ownership", "purpose", "verification_status",
    "term", "emp_length_int", "mths_since_issue_d", "int_rate",
    "mths_since_earliest_cr_line", "acc_now_delinq", "inq_last_6mths",
    "annual_inc", "dti",
]

# Mapping grade -> angka (dipakai untuk membuat risk_score)
GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

DEFAULT_THRESHOLD = 0.5

st.set_page_config(
    page_title="Probability of Default Model",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# LOAD MODEL & METADATA (di-cache supaya tidak reload tiap interaksi)
# ============================================================
@st.cache_resource
def load_model_bundle(path: str):
    """
    Load file pickle model. Mendukung dua format:
    1. Objek model langsung (mis. XGBClassifier)
    2. Dict wrapper, mis. {"model": ..., "threshold": ..., "threshold_metric": ...}

    Return: (model, extra_info_dict)
    """
    if not Path(path).exists():
        return None, {}

    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        # cari key yang paling mungkin berisi model asli
        model_obj = None
        for key in ("model", "estimator", "classifier", "clf", "xgb_model"):
            if key in obj and hasattr(obj[key], "predict_proba"):
                model_obj = obj[key]
                break
        if model_obj is None:
            # fallback: cari value pertama yang punya predict_proba
            for v in obj.values():
                if hasattr(v, "predict_proba"):
                    model_obj = v
                    break
        extra_info = {k: v for k, v in obj.items() if not hasattr(v, "predict_proba")}
        return model_obj, extra_info

    return obj, {}


@st.cache_resource
def load_metadata(path: str):
    if not Path(path).exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def get_expected_features(model, metadata: dict):
    """
    Cari daftar nama fitur final yang diharapkan model, dengan urutan prioritas:
    1. metadata['features'] / metadata['feature_names'] / metadata['selected_features']
    2. model.get_booster().feature_names (khusus XGBoost native/sklearn API)
    3. None -> fallback pakai FEATURE_ENGINEERING manual di bawah
    """
    for key in ("features", "feature_names", "selected_features"):
        if key in metadata and isinstance(metadata[key], list):
            return metadata[key]

    try:
        booster = model.get_booster()
        if booster.feature_names:
            return list(booster.feature_names)
    except Exception:
        pass

    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    except Exception:
        pass

    return None


def get_threshold(metadata: dict, extra_info: dict = None) -> float:
    for source in (extra_info or {}, metadata):
        for key in ("threshold", "optimal_threshold", "cutoff"):
            if key in source:
                try:
                    return float(source[key])
                except Exception:
                    pass
    return DEFAULT_THRESHOLD


# ============================================================
# FEATURE ENGINEERING (sesuai spesifikasi model)
# ============================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # credit history vs employment
    df["credit_emp_ratio"] = df["mths_since_earliest_cr_line"] / (df["emp_length_int"] + 1)

    # risk score dari grade x int_rate
    df["grade_num"] = df["grade"].map(GRADE_MAP)
    df["risk_score"] = df["grade_num"] * df["int_rate"]

    return df


@st.cache_resource
def load_preprocessing_bundle(path: str):
    """
    Load bundle preprocessing hasil training, berisi:
    - scaler: StandardScaler yang sudah di-fit ke data training
    - scaler_columns: daftar kolom numerik yang di-scale
    - label_encoder_home_ownership: LabelEncoder utk home_ownership
    - encoder_column: daftar kolom kategorikal (['home_ownership'])
    - feature_order: urutan kolom final yang diharapkan model
    """
    if not Path(path).exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def preprocess_new_data(df_new: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """
    Replikasi persis fungsi preprocess_new_data dari training pipeline:
    - Encode home_ownership pakai LabelEncoder.transform() (BUKAN fit_transform)
    - Scale kolom numerik pakai StandardScaler.transform() (BUKAN fit_transform)
    - Paksa urutan kolom sesuai feature_order
    """
    df_new = df_new.copy()

    le = bundle["label_encoder_home_ownership"]
    encoder_column = bundle["encoder_column"]
    scaler = bundle["scaler"]
    scaler_columns = bundle["scaler_columns"]
    feature_order = bundle["feature_order"]

    cat_col = encoder_column[0]

    # validasi: tolak kategori yang tidak pernah dilihat model waktu training
    unseen = set(df_new[cat_col].astype(str).unique()) - set(le.classes_)
    if unseen:
        raise ValueError(
            f"Nilai '{cat_col}' berikut tidak dikenali model (tidak ada saat training): "
            f"{sorted(unseen)}. Kategori yang valid: {list(le.classes_)}"
        )

    df_new[cat_col] = le.transform(df_new[cat_col].astype(str))

    # scale numerik - transform saja, pakai mean/std dari training
    X_num_new = pd.DataFrame(
        scaler.transform(df_new[scaler_columns]),
        columns=scaler_columns,
        index=df_new.index,
    )

    X_encoder_new = df_new[encoder_column]
    X_new = pd.concat([X_encoder_new, X_num_new], axis=1)
    X_new = X_new[feature_order]  # paksa urutan kolom sama persis dgn training

    return X_new


def compute_risk_category(pd_value: float) -> str:
    if pd_value >= 0.5:
        return "High"
    elif pd_value >= 0.2:
        return "Medium"
    else:
        return "Low"


RISK_COLORS = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}


def to_native(obj):
    """
    Konversi rekursif tipe numpy/pandas (float32, int64, dst) ke tipe
    native Python supaya bisa di-serialize dengan json.dumps.
    """
    if isinstance(obj, dict):
        return {to_native(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    return obj


def build_portfolio_summary(df: pd.DataFrame, threshold: float) -> dict:
    """Ringkasan statistik portfolio untuk dikirim sebagai konteks ke Claude."""
    total = len(df)
    n_default = int((df["Prediksi"] == "Default").sum())

    summary = {
        "total_customer": total,
        "predicted_default": n_default,
        "default_rate_pct": round(n_default / total * 100, 2),
        "avg_pd_pct": round(float(df["PD"].mean()) * 100, 2),
        "threshold_used": threshold,
        "risk_distribution": df["Risiko"].value_counts().to_dict(),
    }

    if "grade" in df.columns:
        grade_summary = (
            df.groupby("grade")["PD"].mean().sort_index().mul(100).round(2).to_dict()
        )
        summary["avg_pd_by_grade_pct"] = grade_summary

    if "dti" in df.columns:
        summary["avg_dti_pct"] = round(float(df["dti"].mean()), 2)
    if "annual_inc" in df.columns:
        summary["avg_annual_income"] = round(float(df["annual_inc"].mean()), 2)
    if "inq_last_6mths" in df.columns:
        summary["avg_inquiries_6mths"] = round(float(df["inq_last_6mths"].mean()), 2)

    return to_native(summary)


def generate_ai_insight(summary: dict, api_key: str) -> str:
    """Panggil Claude API untuk membuat narasi insight dari ringkasan portfolio."""
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Kamu adalah risk analyst kredit profesional. Lakukan analisis terhadap ringkasan portofolio credit dan berikut ini ringkasan hasil prediksi
probability of default (PD) dari model machine learning untuk sebuah portfolio pinjaman :

{json.dumps(summary, indent=2, default=str)}

Tuliskan analisis naratif singkat (2-3 paragraf pendek, bahasa Indonesia) yang HANYA
membahas dan mejelaskan hasil prediksi di atas, mencakup:
1. Gambaran umum kondisi risiko portfolio (total customer, default rate, avg probability)
2. Pola yang terlihat dari data hasil prediksi (misal hubungan grade dengan PD, distribusi kategori risiko, atau variabel lain yang tersedia di ringkasan)

Aturan penting:
- Bahas HANYA data yang ada di ringkasan JSON di atas, jangan menambahkan poin, asumsi, atau angka yang tidak ada di data.
- JANGAN menuliskan rekomendasi, saran tindakan, atau langkah bisnis apapun.
- Gunakan bahasa profesional namun mudah dipahami, hindari jargon berlebihan.
- Hindari juga membahas terlalu teknis mengenai model, karena insights akan dibaca oleh user yang awam terhadap machine learning modeling."""

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in message.content if block.type == "text")


# ============================================================
# SIDEBAR - INFO MODEL
# ============================================================
model, model_extra_info = load_model_bundle(MODEL_PATH)
metadata = load_metadata(METADATA_PATH)
expected_features = get_expected_features(model, metadata)
threshold = get_threshold(metadata, model_extra_info)
preprocessing_bundle = load_preprocessing_bundle(PREPROCESSING_BUNDLE_PATH)

with st.sidebar:
    st.header("⚙️ Status Model")
    if model is None:
        st.error(f"Model tidak ditemukan / gagal dimuat dari `{MODEL_PATH}`. Pastikan file .pkl valid dan berisi objek model (atau dict yang memuat key 'model').")
    else:
        st.success("Model berhasil dimuat ✅")
        if model_extra_info:
            with st.expander("Info tambahan dari file pickle"):
                st.json({k: v for k, v in model_extra_info.items() if isinstance(v, (str, int, float, bool, list, dict))})

    if not metadata:
        st.warning(f"Metadata tidak ditemukan di `{METADATA_PATH}`.")
    else:
        st.success("Metadata berhasil dimuat ✅")

    if preprocessing_bundle is None:
        st.error(
            f"Preprocessing bundle tidak ditemukan di `{PREPROCESSING_BUNDLE_PATH}`. "
            "Prediksi tidak bisa dilakukan tanpa ini (scaler & label encoder wajib)."
        )
    else:
        st.success("Preprocessing bundle (scaler + label encoder) berhasil dimuat ✅")
        with st.expander("Detail preprocessing bundle"):
            st.write("Kolom di-scale:", preprocessing_bundle.get("scaler_columns"))
            st.write("Kolom di-encode:", preprocessing_bundle.get("encoder_column"))
            le = preprocessing_bundle.get("label_encoder_home_ownership")
            if le is not None:
                st.write("Kategori home_ownership yang dikenal:", list(le.classes_))
            st.write("Urutan fitur final:", preprocessing_bundle.get("feature_order"))

    st.write(f"**Threshold klasifikasi:** {threshold:.3f}")
    threshold_metric = metadata.get("threshold_metric") or model_extra_info.get("threshold_metric")
    if threshold_metric:
        st.caption(f"Threshold dioptimasi berdasarkan metrik: {str(threshold_metric).upper()}")
    if expected_features:
        with st.expander("Lihat fitur final model"):
            st.write(expected_features)

    if not ANTHROPIC_AVAILABLE:
        st.warning("Package `anthropic` belum terinstall. Jalankan `pip install anthropic`.")
    def _get_secret_key():
        try:
            return st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            return ""

    api_key_input = _get_secret_key()

   # ------------------------------------------------------------
    # Credit / footer di pojok kiri bawah sidebar
    # ------------------------------------------------------------
    st.markdown(
        """
        <div style="position: fixed; bottom: 12px; left: 18px; opacity: 0.6; font-size: 0.8rem;">
            created by Ashari Retno
        </div>
        """,
        unsafe_allow_html=True,
    )
 

# ============================================================
# HEADER
# ============================================================
st.title("📊 Probability of Default (PD) Model")
st.caption("Upload data customer untuk memprediksi risiko gagal bayar menggunakan model XGBoost.")

# ============================================================
# UPLOAD / SAMPLE DATA
# ============================================================
col_upload, col_sample = st.columns([2, 1])

uploaded_file = col_upload.file_uploader(
    "Drop file CSV di sini atau klik untuk pilih file",
    type=["csv"],
    help=f"Kolom yang dibutuhkan: {', '.join(RAW_REQUIRED_COLUMNS)}",
)

use_sample = col_sample.button("📄 Coba dengan sample data (20 customers)", use_container_width=True)

df_raw = None

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
elif use_sample:
    sample_path = Path("sample_data.csv")
    if sample_path.exists():
        df_raw = pd.read_csv(sample_path)
    else:
        st.error("File sample_data.csv tidak ditemukan di folder app.")

# ============================================================
# PROSES PREDIKSI
# ============================================================
if df_raw is not None:
    missing_cols = [c for c in RAW_REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing_cols:
        st.error(f"Kolom berikut tidak ditemukan di file kamu: {missing_cols}")
        st.stop()

    if model is None:
        st.error("Model belum tersedia, tidak bisa melakukan prediksi.")
        st.stop()

    if preprocessing_bundle is None:
        st.error(
            f"Preprocessing bundle (`{PREPROCESSING_BUNDLE_PATH}`) belum tersedia. "
            "Ini wajib ada karena model dilatih dari data yang sudah di-scaling & di-encode. "
            "Upload file bundle-nya ke folder app ini."
        )
        st.stop()

    st.success(f"{uploaded_file.name if uploaded_file else 'sample_data.csv'} berhasil diproses ({len(df_raw)} baris).")

    # --- feature engineering ---
    df_fe = engineer_features(df_raw)

    # --- preprocessing: label encode home_ownership + scale numerik, transform-only ---
    try:
        X = preprocess_new_data(df_fe, preprocessing_bundle)
    except ValueError as e:
        st.error(f"Gagal melakukan preprocessing data.\n\n{e}")
        st.stop()
    except Exception as e:
        st.error(f"Gagal melakukan preprocessing data. Detail error: {e}")
        st.stop()

    # --- prediksi per baris ---
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception as e:
        st.error(
            "Gagal melakukan prediksi. Kemungkinan penyebab: tipe data kolom tidak "
            f"sesuai dengan yang diharapkan model.\n\nDetail error: {e}"
        )
        st.stop()
    df_raw["PD"] = proba
    df_raw["Prediksi"] = np.where(df_raw["PD"] >= threshold, "Default", "Non-default")
    df_raw["Risiko"] = df_raw["PD"].apply(compute_risk_category)

    # ============================================================
    # METRICS
    # ============================================================
    total_customer = len(df_raw)
    predicted_default = int((df_raw["Prediksi"] == "Default").sum())
    default_rate = predicted_default / total_customer * 100
    avg_proba = df_raw["PD"].mean() * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total customer", f"{total_customer:,}".replace(",", "."))
    m2.metric("Predicted default", f"{predicted_default:,}".replace(",", "."))
    m3.metric("Default rate", f"{default_rate:.1f}%")
    m4.metric("Avg probability", f"{avg_proba:.1f}%")

    # ============================================================
    # CHARTS
    # ============================================================
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Breakdown prediksi model")
        pred_counts = df_raw["Prediksi"].value_counts()
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=pred_counts.index,
                    values=pred_counts.values,
                    hole=0.6,
                    marker=dict(colors=["#ef4444", "#22c55e"]),
                )
            ]
        )
        fig_donut.update_layout(
            showlegend=True,
            margin=dict(t=10, b=10, l=10, r=10),
            height=350,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        st.subheader("Distribusi kategori risiko")
        risk_counts = df_raw["Risiko"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        fig_bar = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            color=risk_counts.index,
            color_discrete_map=RISK_COLORS,
        )
        fig_bar.update_layout(
            showlegend=False,
            xaxis_title=None,
            yaxis_title=None,
            margin=dict(t=10, b=10, l=10, r=10),
            height=350,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ============================================================
    # TABEL HASIL
    # ============================================================
    st.subheader("Hasil prediksi per customer")

    display_df = df_raw[["loan_id", "grade", "int_rate", "dti", "PD", "Risiko"]].copy()
    display_df.columns = ["Loan ID", "Grade", "Int rate", "Dti", "PD", "Risiko"]
    display_df["Int rate"] = display_df["Int rate"].map(lambda x: f"{x:.1f}%")
    display_df["Dti"] = display_df["Dti"].map(lambda x: f"{x:.1f}%")
    display_df["PD"] = display_df["PD"].map(lambda x: f"{x*100:.1f}%")

    def highlight_risk(val):
        color = RISK_COLORS.get(val, "")
        return f"color: {color}; font-weight: 600" if color else ""

    styler = display_df.style
    # pandas >= 2.1 pakai .map, versi lebih lama pakai .applymap (deprecated)
    if hasattr(styler, "map"):
        styler = styler.map(highlight_risk, subset=["Risiko"])
    else:
        styler = styler.applymap(highlight_risk, subset=["Risiko"])

    st.dataframe(
        styler,
        use_container_width=True,
        height=350,
    )
    st.caption(f"Total {len(display_df)} baris ditampilkan.")

    # ============================================================
    # AI GENERATED INSIGHT
    # ============================================================
    st.divider()
    st.subheader("✨ AI generated insight")

    if "ai_insight_text" not in st.session_state:
        st.session_state.ai_insight_text = None

    with st.container(border=True):
        st.caption("Klik tombol di bawah untuk membuat Claude menganalisis hasil prediksi portfolio ini.")
        gen_clicked = st.button("✨ Generate AI insight", type="primary", use_container_width=True)

        if gen_clicked:
            if not ANTHROPIC_AVAILABLE:
                st.error("Package `anthropic` belum terinstall di environment ini. Jalankan: `pip install anthropic`")
            elif not api_key_input:
                st.error("Masukkan Anthropic API Key di sidebar terlebih dahulu.")
            else:
                with st.spinner("Claude sedang menganalisis portfolio..."):
                    try:
                        summary = build_portfolio_summary(df_raw, threshold)
                        insight_text = generate_ai_insight(summary, api_key_input)
                        st.session_state.ai_insight_text = insight_text
                    except Exception as e:
                        st.error(f"Gagal generate insight: {e}")

        if st.session_state.ai_insight_text:
            st.markdown(st.session_state.ai_insight_text)

    # ============================================================
    # DOWNLOAD HASIL
    # ============================================================
    csv_result = df_raw.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download hasil prediksi (CSV)",
        data=csv_result,
        file_name="hasil_prediksi_pd.csv",
        mime="text/csv",
    )

else:
    st.info("Silakan upload file CSV atau gunakan sample data untuk memulai prediksi.")