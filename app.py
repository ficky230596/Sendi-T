
import html
import io

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from folium.plugins import MarkerCluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from streamlit_folium import st_folium


# =========================================================
# 1. KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="Klasterisasi Kriminalitas Jayawijaya",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Sistem Klasterisasi Tingkat Kriminalitas Kabupaten Jayawijaya")

st.markdown(
    """
    Aplikasi untuk mengelompokkan lokasi berdasarkan jumlah kasus
    pada setiap jenis kejahatan menggunakan algoritma **K-Means Clustering**.

    Hasil pengelompokan dianalisis menggunakan **Elbow Method** dan
    **Silhouette Score**. Peta hanya menampilkan penanda titik lokasi,
    bukan pewarnaan batas wilayah.
    """
)


# =========================================================
# 2. FUNGSI BANTU
# =========================================================

REQUIRED_COLS = {
    "Alamat/Lokasi Kejadian",
    "Jenis Kejahatan",
    "Jumlah Kasus"
}

MAP_COLORS = [
    "red",
    "blue",
    "green",
    "purple",
    "orange",
    "darkred",
    "cadetblue",
    "darkgreen",
    "darkpurple",
    "darkblue"
]


def read_dataset(uploaded_file):
    """Membaca dataset Excel atau CSV."""
    if uploaded_file.name.lower().endswith(".csv"):
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file), "CSV"

    uploaded_file.seek(0)
    excel_file = pd.ExcelFile(uploaded_file)

    if "Rekap Alamat" in excel_file.sheet_names:
        sheet_name = "Rekap Alamat"
    else:
        sheet_name = excel_file.sheet_names[0]

    data = pd.read_excel(
        uploaded_file,
        sheet_name=sheet_name
    )

    return data, sheet_name


def calculate_silhouette(X_scaled, labels):
    """
    Menghitung Silhouette Score jika syarat evaluasi terpenuhi.
    Nilai ini mengevaluasi pemisahan klaster, bukan akurasi klasifikasi.
    """
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return None

    if len(unique_labels) >= len(labels):
        return None

    try:
        return float(silhouette_score(X_scaled, labels))
    except Exception:
        return None


def make_download_csv(dataframe):
    """Membuat data CSV dengan dukungan karakter UTF-8."""
    return dataframe.to_csv(index=False).encode("utf-8-sig")


# =========================================================
# 3. UPLOAD DATASET
# =========================================================

st.sidebar.header("📁 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload dataset Excel atau CSV",
    type=["xlsx", "csv"]
)

df_rekap = None
source_name = ""

if uploaded_file is not None:
    try:
        df_rekap, source_name = read_dataset(uploaded_file)

        st.sidebar.success(
            f"Dataset berhasil dibaca: {source_name}"
        )

    except Exception as e:
        st.error(f"Gagal membaca dataset: {e}")
        st.stop()

else:
    st.info(
        "Aplikasi akan mencoba membaca dataset default. "
        "Anda juga dapat mengunggah file melalui sidebar."
    )

    default_file = "Data_Kriminal_Jayawijaya_Berdasarkan_Alamat.xlsx"

    try:
        excel_file = pd.ExcelFile(default_file)

        if "Rekap Alamat" in excel_file.sheet_names:
            sheet_name = "Rekap Alamat"
        else:
            sheet_name = excel_file.sheet_names[0]

        df_rekap = pd.read_excel(
            default_file,
            sheet_name=sheet_name
        )

        source_name = f"{default_file} — sheet {sheet_name}"

    except FileNotFoundError:
        st.warning(
            "Dataset default tidak ditemukan. "
            "Silakan unggah file Excel atau CSV melalui sidebar."
        )
        st.stop()

    except Exception as e:
        st.error(f"Gagal membaca dataset default: {e}")
        st.stop()


if df_rekap is None or df_rekap.empty:
    st.error("Dataset kosong atau tidak berhasil dimuat.")
    st.stop()

st.caption(f"Sumber file yang dibaca: {source_name}")


# =========================================================
# 4. VALIDASI DAN PEMBERSIHAN DATA
# =========================================================

df_rekap = df_rekap.copy()

# Membersihkan nama kolom dari spasi yang tidak disengaja.
df_rekap.columns = [
    str(col).strip()
    for col in df_rekap.columns
]

missing_cols = REQUIRED_COLS - set(df_rekap.columns)

if missing_cols:
    st.error(
        "Kolom dataset belum lengkap: "
        + ", ".join(sorted(missing_cols))
    )

    st.write("Kolom yang tersedia:", df_rekap.columns.tolist())

    st.info(
        "Pastikan kolom utama bernama "
        "'Alamat/Lokasi Kejadian', 'Jenis Kejahatan', dan 'Jumlah Kasus'."
    )

    st.stop()


# Simpan jumlah data awal untuk dokumentasi pembersihan.
initial_row_count = len(df_rekap)

# Bersihkan kolom teks.
for col in ["Alamat/Lokasi Kejadian", "Jenis Kejahatan"]:
    df_rekap[col] = (
        df_rekap[col]
        .astype("string")
        .str.strip()
    )

    df_rekap[col] = df_rekap[col].replace(
        ["", "nan", "NaN", "<NA>", "None"],
        pd.NA
    )

# Konversi jumlah kasus menjadi angka.
# Data yang tidak bisa dikonversi akan menjadi NaN dan dihapus.
df_rekap["Jumlah Kasus"] = pd.to_numeric(
    df_rekap["Jumlah Kasus"],
    errors="coerce"
)

# Hapus baris yang tidak memiliki data inti.
df_rekap = df_rekap.dropna(
    subset=[
        "Alamat/Lokasi Kejadian",
        "Jenis Kejahatan",
        "Jumlah Kasus"
    ]
).copy()

# Jumlah kasus harus berupa bilangan bulat nonnegatif.
df_rekap = df_rekap[
    (df_rekap["Jumlah Kasus"] >= 0)
    & (df_rekap["Jumlah Kasus"] % 1 == 0)
].copy()

if df_rekap.empty:
    st.error(
        "Tidak ada data valid setelah pembersihan. "
        "Periksa kembali lokasi, jenis kejahatan, dan jumlah kasus."
    )
    st.stop()

df_rekap["Jumlah Kasus"] = df_rekap["Jumlah Kasus"].astype("int64")

# Informasi koordinat bersifat opsional.
has_lat = "Latitude" in df_rekap.columns
has_lon = "Longitude" in df_rekap.columns
has_coordinates = has_lat and has_lon

if has_coordinates:
    # Ubah koordinat menjadi numerik.
    df_rekap["Latitude"] = pd.to_numeric(
        df_rekap["Latitude"],
        errors="coerce"
    )

    df_rekap["Longitude"] = pd.to_numeric(
        df_rekap["Longitude"],
        errors="coerce"
    )

    valid_coords = (
        df_rekap["Latitude"].between(-90, 90)
        & df_rekap["Longitude"].between(-180, 180)
    )

    # Koordinat yang tidak valid tidak digunakan pada peta.
    df_rekap.loc[
        ~valid_coords,
        ["Latitude", "Longitude"]
    ] = np.nan

elif has_lat or has_lon:
    st.warning(
        "Dataset hanya memiliki salah satu kolom koordinat. "
        "Peta membutuhkan Latitude dan Longitude sekaligus."
    )


# =========================================================
# 5. INFORMASI KUALITAS DATA
# =========================================================

st.subheader("🔎 Pemeriksaan Kualitas Data")

valid_row_count = len(df_rekap)
removed_row_count = initial_row_count - valid_row_count

quality_col1, quality_col2, quality_col3 = st.columns(3)

quality_col1.metric(
    "Baris Data Awal",
    f"{initial_row_count:,}"
)

quality_col2.metric(
    "Baris Data Valid",
    f"{valid_row_count:,}"
)

quality_col3.metric(
    "Baris Dihapus",
    f"{removed_row_count:,}"
)

with st.expander("Lihat data setelah pembersihan"):
    st.dataframe(
        df_rekap,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 6. AGREGASI DATA BERDASARKAN LOKASI DAN JENIS KEJAHATAN
# =========================================================

# Setiap baris hasil pivot mewakili satu lokasi.
# Setiap kolom fitur mewakili satu jenis kejahatan.
pivot_df = df_rekap.pivot_table(
    index="Alamat/Lokasi Kejadian",
    columns="Jenis Kejahatan",
    values="Jumlah Kasus",
    aggfunc="sum",
    fill_value=0
)

pivot_df.columns.name = None

if pivot_df.empty:
    st.error("Tidak ada lokasi yang dapat dianalisis.")
    st.stop()

crime_cols = pivot_df.columns.tolist()

if not crime_cols:
    st.error("Tidak ditemukan jenis kejahatan untuk analisis.")
    st.stop()

pivot_df["Total Kasus"] = pivot_df[crime_cols].sum(axis=1)

# Tambahkan koordinat representatif untuk setiap lokasi.
# Jika satu lokasi mempunyai beberapa koordinat valid,
# aplikasi menggunakan pasangan valid pertama.
if has_coordinates:
    coord_df = df_rekap.dropna(
        subset=["Latitude", "Longitude"]
    ).copy()

    coord_df = coord_df.drop_duplicates(
        subset=["Alamat/Lokasi Kejadian"],
        keep="first"
    ).set_index("Alamat/Lokasi Kejadian")

    pivot_df["Latitude"] = coord_df["Latitude"]
    pivot_df["Longitude"] = coord_df["Longitude"]

    # Koordinat tidak tersedia untuk lokasi tertentu akan tetap NaN.
    pivot_df["Latitude"] = pd.to_numeric(
        pivot_df["Latitude"],
        errors="coerce"
    )

    pivot_df["Longitude"] = pd.to_numeric(
        pivot_df["Longitude"],
        errors="coerce"
    )

number_of_locations = len(pivot_df)

if number_of_locations < 3:
    st.error(
        "Analisis klaster dan evaluasi Silhouette membutuhkan "
        "jumlah lokasi yang memadai. Dataset ini harus memiliki "
        "minimal 3 lokasi berbeda."
    )
    st.stop()


# =========================================================
# 7. PEMILIHAN FITUR DAN VALIDASI K-MEANS
# =========================================================

X = pivot_df[crime_cols].astype(float)

# Periksa apakah ada setidaknya dua pola data yang berbeda.
unique_feature_rows = np.unique(X.to_numpy(), axis=0).shape[0]

if unique_feature_rows < 2:
    st.error(
        "Semua lokasi memiliki pola jumlah kasus yang sama. "
        "K-Means tidak dapat membentuk kelompok yang bermakna. "
        "Periksa data atau tambahkan data lokasi yang valid."
    )
    st.stop()

# Standardisasi fitur agar skala fitur lebih sebanding.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Batas K dipilih agar:
# - K minimal 2
# - K tidak melebihi jumlah lokasi
# - K tidak melebihi jumlah pola fitur unik
max_k_allowed = min(
    5,
    number_of_locations - 1,
    unique_feature_rows
)

if max_k_allowed < 2:
    st.error(
        "Jumlah lokasi atau variasi fitur tidak cukup "
        "untuk menjalankan K-Means."
    )
    st.stop()


# =========================================================
# 8. PENGATURAN MODEL
# =========================================================

st.sidebar.header("⚙️ Pengaturan Model K-Means")

k = st.sidebar.slider(
    "Jumlah Klaster (K)",
    min_value=2,
    max_value=max_k_allowed,
    value=min(3, max_k_allowed),
    help=(
        "Nilai K menentukan jumlah kelompok yang dibentuk. "
        "Gunakan hasil evaluasi model dan pertimbangan penelitian "
        "untuk menentukan K yang akan digunakan."
    )
)

random_state = st.sidebar.number_input(
    "Random State",
    min_value=0,
    max_value=2147483647,
    value=42,
    step=1,
    help="Seed agar hasil dapat direproduksi dengan pengaturan yang sama."
)


# =========================================================
# 9. ELBOW METHOD DAN SILHOUETTE SCORE
# =========================================================

st.subheader("📐 Evaluasi Jumlah Klaster")

st.markdown(
    """
    **Elbow Method** membantu melihat perubahan inertia ketika K bertambah.
    **Silhouette Score** menilai seberapa baik suatu lokasi berada dalam
    klasternya dibandingkan dengan klaster lain. Nilainya berkisar dari
    -1 sampai 1; nilai yang lebih tinggi umumnya menunjukkan pemisahan
    klaster yang lebih baik.
    """
)

max_eval_k = min(
    8,
    number_of_locations - 1,
    unique_feature_rows
)

evaluation_rows = []
inertia_rows = []

for candidate_k in range(1, max_eval_k + 1):
    candidate_model = KMeans(
        n_clusters=candidate_k,
        random_state=int(random_state),
        n_init=20
    )

    candidate_labels = candidate_model.fit_predict(X_scaled)

    inertia_rows.append({
        "K": candidate_k,
        "Inertia": float(candidate_model.inertia_)
    })

    if candidate_k >= 2:
        score = calculate_silhouette(
            X_scaled,
            candidate_labels
        )

        evaluation_rows.append({
            "K": candidate_k,
            "Silhouette Score": score,
            "Inertia": float(candidate_model.inertia_),
            "Jumlah Klaster Terbentuk": len(
                np.unique(candidate_labels)
            )
        })

evaluation_df = pd.DataFrame(evaluation_rows)
inertia_df = pd.DataFrame(inertia_rows)

eval_col1, eval_col2 = st.columns(2)

with eval_col1:
    st.markdown("**Elbow Method (Inertia)**")

    fig_elbow, ax_elbow = plt.subplots(figsize=(7, 4))

    ax_elbow.plot(
        inertia_df["K"],
        inertia_df["Inertia"],
        marker="o"
    )

    ax_elbow.set_xlabel("Jumlah Klaster (K)")
    ax_elbow.set_ylabel("Inertia")
    ax_elbow.set_title("Elbow Method")
    ax_elbow.set_xticks(inertia_df["K"])
    ax_elbow.grid(True, linestyle="--", alpha=0.4)

    fig_elbow.tight_layout()
    st.pyplot(fig_elbow)
    plt.close(fig_elbow)

with eval_col2:
    st.markdown("**Silhouette Score**")

    valid_silhouette_df = evaluation_df.dropna(
        subset=["Silhouette Score"]
    )

    if not valid_silhouette_df.empty:
        fig_sil, ax_sil = plt.subplots(figsize=(7, 4))

        ax_sil.plot(
            valid_silhouette_df["K"],
            valid_silhouette_df["Silhouette Score"],
            marker="o"
        )

        ax_sil.set_xlabel("Jumlah Klaster (K)")
        ax_sil.set_ylabel("Silhouette Score")
        ax_sil.set_title("Evaluasi Silhouette Score")
        ax_sil.set_xticks(valid_silhouette_df["K"])
        ax_sil.grid(True, linestyle="--", alpha=0.4)

        fig_sil.tight_layout()
        st.pyplot(fig_sil)
        plt.close(fig_sil)

    else:
        st.warning(
            "Silhouette Score tidak dapat dihitung untuk "
            "kombinasi klaster yang tersedia."
        )

st.markdown("**Tabel evaluasi kandidat K**")

evaluation_display = evaluation_df.copy()

evaluation_display["Silhouette Score"] = (
    evaluation_display["Silhouette Score"].round(4)
)

evaluation_display["Inertia"] = (
    evaluation_display["Inertia"].round(4)
)

st.dataframe(
    evaluation_display,
    use_container_width=True,
    hide_index=True
)

if not valid_silhouette_df.empty:
    best_row = valid_silhouette_df.loc[
        valid_silhouette_df["Silhouette Score"].idxmax()
    ]

    st.info(
        f"Berdasarkan Silhouette Score pada kandidat yang diuji, "
        f"K dengan nilai tertinggi adalah {int(best_row['K'])} "
        f"(Silhouette Score = "
        f"{best_row['Silhouette Score']:.4f}). "
        f"Pertimbangkan juga Elbow Method, karakteristik data, "
        f"dan tujuan penelitian sebelum menetapkan K akhir."
    )


# =========================================================
# 10. MODEL K-MEANS FINAL
# =========================================================

kmeans = KMeans(
    n_clusters=int(k),
    random_state=int(random_state),
    n_init=20
)

raw_labels = kmeans.fit_predict(X_scaled)

# Simpan ID klaster asli yang dibuat K-Means.
pivot_df["ID Klaster K-Means"] = raw_labels

# Urutkan klaster berdasarkan rata-rata total kasus.
cluster_means = (
    pivot_df.groupby("ID Klaster K-Means")["Total Kasus"]
    .mean()
    .sort_values(ascending=False, kind="stable")
)

ranked_cluster_ids = cluster_means.index.tolist()

# ID peringkat dibuat terpisah dari ID asli K-Means.
rank_map = {
    cluster_id: rank + 1
    for rank, cluster_id in enumerate(ranked_cluster_ids)
}

pivot_df["Peringkat Klaster"] = (
    pivot_df["ID Klaster K-Means"].map(rank_map)
)

if k == 3:
    level_map = {
        ranked_cluster_ids[0]: "Tinggi",
        ranked_cluster_ids[1]: "Sedang",
        ranked_cluster_ids[2]: "Rendah"
    }

    pivot_df["Tingkat Kriminalitas"] = (
        pivot_df["ID Klaster K-Means"].map(level_map)
    )

else:
    # Jika K bukan 3, hindari menyebut hasilnya Tinggi/Sedang/Rendah.
    label_map = {
        cluster_id: f"Klaster {rank + 1}"
        for rank, cluster_id in enumerate(ranked_cluster_ids)
    }

    pivot_df["Tingkat Kriminalitas"] = (
        pivot_df["ID Klaster K-Means"].map(label_map)
    )

# Warna marker mengikuti kategori atau peringkat klaster.
if k == 3:
    color_map = {
        "Tinggi": "red",
        "Sedang": "orange",
        "Rendah": "green"
    }
else:
    color_map = {
        f"Klaster {rank + 1}": MAP_COLORS[rank % len(MAP_COLORS)]
        for rank in range(k)
    }

final_silhouette = calculate_silhouette(
    X_scaled,
    raw_labels
)


# =========================================================
# 11. RINGKASAN HASIL ANALISIS
# =========================================================

st.subheader("📊 Ringkasan Hasil Analisis")

total_locations = len(pivot_df)
total_cases = int(pivot_df["Total Kasus"].sum())

sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)

sum_col1.metric(
    "Lokasi Dianalisis",
    f"{total_locations:,}"
)

sum_col2.metric(
    "Total Kasus",
    f"{total_cases:,}"
)

sum_col3.metric(
    "Jumlah Klaster (K)",
    int(k)
)

if final_silhouette is not None:
    sum_col4.metric(
        "Silhouette Score",
        f"{final_silhouette:.4f}"
    )
else:
    sum_col4.metric(
        "Silhouette Score",
        "Tidak tersedia"
    )

st.caption(
    "Total kasus merupakan penjumlahan berdasarkan data yang telah "
    "dibersihkan dan diagregasi per lokasi. Pastikan sumber data tidak "
    "mengandung baris duplikat yang menyebabkan penghitungan ganda."
)


# =========================================================
# 12. RINGKASAN KARAKTERISTIK KLASTER
# =========================================================

st.subheader("🧮 Karakteristik Setiap Klaster")

cluster_summary = (
    pivot_df.groupby(
        ["Peringkat Klaster", "Tingkat Kriminalitas"],
        as_index=False
    )
    .agg(
        Jumlah_Lokasi=("Total Kasus", "count"),
        Total_Kasus=("Total Kasus", "sum"),
        Rata_Rata_Kasus=("Total Kasus", "mean"),
        Minimum_Kasus=("Total Kasus", "min"),
        Maksimum_Kasus=("Total Kasus", "max")
    )
    .sort_values("Peringkat Klaster")
)

cluster_summary = cluster_summary.rename(
    columns={
        "Jumlah_Lokasi": "Jumlah Lokasi",
        "Total_Kasus": "Total Kasus",
        "Rata_Rata_Kasus": "Rata-rata Kasus per Lokasi",
        "Minimum_Kasus": "Kasus Minimum",
        "Maksimum_Kasus": "Kasus Maksimum"
    }
)

cluster_summary["Rata-rata Kasus per Lokasi"] = (
    cluster_summary["Rata-rata Kasus per Lokasi"].round(2)
)

st.dataframe(
    cluster_summary,
    use_container_width=True,
    hide_index=True
)

if k == 3:
    st.caption(
        "Label Tinggi, Sedang, dan Rendah ditentukan dari rata-rata "
        "total kasus tiap klaster. Label ini adalah interpretasi hasil "
        "pengelompokan, bukan kategori resmi dari instansi kepolisian."
    )
else:
    st.caption(
        "Karena K bukan 3, hasil ditampilkan sebagai Klaster 1, Klaster 2, "
        "dan seterusnya, diurutkan berdasarkan rata-rata total kasus "
        "dari tertinggi ke terendah."
    )


# =========================================================
# 13. FILTER LOKASI
# =========================================================

st.sidebar.header("🔎 Filter Tampilan")

search_location = st.sidebar.text_input(
    "Cari nama lokasi",
    value="",
    help="Filter tabel dan penanda peta berdasarkan nama lokasi."
)

if search_location.strip():
    filtered_df = pivot_df[
        pivot_df.index.to_series()
        .str.contains(
            search_location.strip(),
            case=False,
            na=False,
            regex=False
        )
    ].copy()
else:
    filtered_df = pivot_df.copy()

st.caption(
    f"Lokasi yang cocok dengan filter: {len(filtered_df)} "
    f"dari {len(pivot_df)} lokasi."
)


# =========================================================
# 14. PETA PENANDA TITIK LOKASI
# =========================================================

st.subheader("📍 Peta Lokasi Kriminalitas")

st.caption(
    "Peta menampilkan marker titik lokasi. Warna marker menunjukkan "
    "kategori hasil klasterisasi, bukan tingkat risiko resmi atau "
    "sebaran kejadian yang presisi. Pastikan koordinat berasal dari "
    "sumber yang dapat dipertanggungjawabkan."
)

if not has_coordinates:
    st.warning(
        "Peta tidak dapat menampilkan marker karena dataset belum "
        "memiliki kolom Latitude dan Longitude. Tambahkan kedua kolom "
        "dengan koordinat lokasi yang telah diverifikasi."
    )

else:
    map_df = filtered_df.dropna(
        subset=["Latitude", "Longitude"]
    ).copy()

    if map_df.empty:
        st.warning(
            "Tidak ada lokasi dengan koordinat valid untuk ditampilkan. "
            "Periksa kolom Latitude dan Longitude serta filter lokasi."
        )

    else:
        center_lat = float(map_df["Latitude"].mean())
        center_lon = float(map_df["Longitude"].mean())

        crime_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            control_scale=True
        )

        marker_cluster = MarkerCluster().add_to(crime_map)

        for location, row in map_df.iterrows():
            level = str(row["Tingkat Kriminalitas"])
            marker_color = color_map.get(level, "blue")

            safe_location = html.escape(str(location))
            safe_level = html.escape(level)

            popup_html = f"""
            <div style="width:250px">
                <h4>{safe_location}</h4>
                <b>Total kasus:</b> {int(row['Total Kasus'])}<br>
                <b>Tingkat/kelompok:</b> {safe_level}<br>
                <b>ID Klaster K-Means:</b>
                    {int(row['ID Klaster K-Means']) + 1}<br>
                <b>Koordinat:</b><br>
                {float(row['Latitude']):.6f},
                {float(row['Longitude']):.6f}
            </div>
            """

            folium.Marker(
                location=[
                    float(row["Latitude"]),
                    float(row["Longitude"])
                ],
                tooltip=str(location),
                popup=folium.Popup(
                    popup_html,
                    max_width=320
                ),
                icon=folium.Icon(
                    color=marker_color,
                    icon="info-sign"
                )
            ).add_to(marker_cluster)

        st_folium(
            crime_map,
            use_container_width=True,
            height=550,
            key="peta_kriminalitas"
        )

        st.caption(
            f"Menampilkan {len(map_df)} lokasi berkoordinat valid "
            f"dari {len(filtered_df)} lokasi hasil filter."
        )

        if len(map_df) < len(filtered_df):
            st.warning(
                "Sebagian lokasi tidak memiliki koordinat valid "
                "dan tidak ditampilkan pada peta."
            )


# =========================================================
# 15. TABEL HASIL KLASTERISASI
# =========================================================

st.subheader("📋 Tabel Hasil Klasterisasi per Lokasi")

display_cols = [
    "Total Kasus",
    "ID Klaster K-Means",
    "Peringkat Klaster",
    "Tingkat Kriminalitas"
] + crime_cols

result_df = (
    filtered_df[display_cols]
    .sort_values("Total Kasus", ascending=False)
    .reset_index()
)

st.dataframe(
    result_df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 16. VISUALISASI SEBARAN KLASTER
# =========================================================

st.subheader("📈 Visualisasi Sebaran Klaster")

# PCA digunakan hanya untuk memvisualisasikan data berdimensi banyak
# dalam dua dimensi; PCA tidak mengubah fitur yang dipakai K-Means.
if len(crime_cols) >= 2 and len(pivot_df) >= 2:
    pca = PCA(n_components=2)

    pca_components = pca.fit_transform(X_scaled)

    plot_df = pd.DataFrame(
        {
            "Komponen Utama 1": pca_components[:, 0],
            "Komponen Utama 2": pca_components[:, 1],
            "Tingkat Kriminalitas": pivot_df[
                "Tingkat Kriminalitas"
            ].values
        },
        index=pivot_df.index
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=plot_df,
        x="Komponen Utama 1",
        y="Komponen Utama 2",
        hue="Tingkat Kriminalitas",
        palette="tab10",
        s=120,
        edgecolor="black",
        ax=ax
    )

    ax.set_title(
        f"Visualisasi PCA Hasil K-Means (K={k})"
    )

    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    explained_variance = float(
        pca.explained_variance_ratio_.sum() * 100
    )

    st.caption(
        f"Dua komponen PCA menjelaskan "
        f"{explained_variance:.2f}% variasi data yang distandardisasi. "
        "Grafik ini merupakan visualisasi dua dimensi; model K-Means "
        "tetap dilatih menggunakan seluruh fitur jenis kejahatan."
    )

else:
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=pivot_df,
        x=crime_cols[0],
        y="Total Kasus",
        hue="Tingkat Kriminalitas",
        palette="tab10",
        s=120,
        edgecolor="black",
        ax=ax
    )

    ax.set_title(
        f"Sebaran Kasus dan Klaster (K={k})"
    )

    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# 17. DISTRIBUSI JUMLAH LOKASI PER KLASTER
# =========================================================

st.subheader("📊 Distribusi Lokasi per Klaster")

cluster_counts = (
    pivot_df["Tingkat Kriminalitas"]
    .value_counts()
    .rename_axis("Tingkat Kriminalitas")
    .reset_index(name="Jumlah Lokasi")
)

st.bar_chart(
    cluster_counts.set_index("Tingkat Kriminalitas")
)


# =========================================================
# 18. DOWNLOAD HASIL ANALISIS
# =========================================================

st.subheader("📥 Unduh Hasil Analisis")

# Seluruh lokasi disertakan pada file hasil, tidak hanya lokasi yang
# sedang ditampilkan melalui filter pencarian.
full_result_df = (
    pivot_df.reset_index()
    .sort_values("Total Kasus", ascending=False)
)

st.download_button(
    label="📥 Download Hasil Klasterisasi (CSV)",
    data=make_download_csv(full_result_df),
    file_name="hasil_klasterisasi_kriminalitas_jayawijaya.csv",
    mime="text/csv"
)

st.download_button(
    label="📥 Download Evaluasi Model (CSV)",
    data=make_download_csv(evaluation_df),
    file_name="evaluasi_kmeans_jayawijaya.csv",
    mime="text/csv"
)

st.download_button(
    label="📥 Download Ringkasan Klaster (CSV)",
    data=make_download_csv(cluster_summary),
    file_name="ringkasan_klaster_jayawijaya.csv",
    mime="text/csv"
)


# =========================================================
# 19. CATATAN METODOLOGI
# =========================================================

with st.expander("📚 Catatan Metodologi untuk Laporan Penelitian"):
    st.markdown(
        """
        **Unit analisis**

        Satu baris pada matriks fitur mewakili satu lokasi. Kolom fitur
        merupakan jenis kejahatan, sedangkan nilainya merupakan jumlah
        kasus untuk kombinasi lokasi dan jenis kejahatan tersebut.

        **Prapemrosesan**

        Data inti yang kosong atau tidak valid dihapus. Jumlah kasus
        dikonversi menjadi angka bulat nonnegatif. Data kemudian
        diagregasi berdasarkan lokasi dan jenis kejahatan.

        **Standardisasi**

        Fitur jumlah kasus distandardisasi menggunakan `StandardScaler`
        sebelum proses K-Means.

        **Klasterisasi**

        K-Means dijalankan dengan `n_init=20` dan nilai `random_state`
        yang dapat diatur untuk membantu reproduksibilitas hasil.

        **Evaluasi**

        Elbow Method digunakan untuk mengamati perubahan inertia.
        Silhouette Score digunakan untuk menilai pemisahan klaster.
        Tidak satu pun dari kedua metrik tersebut merupakan akurasi
        klasifikasi berlabel.

        **Interpretasi**

        Untuk K=3, nama Tinggi, Sedang, dan Rendah ditetapkan berdasarkan
        urutan rata-rata total kasus per klaster. Penamaan ini merupakan
        interpretasi analitis terhadap dataset, bukan penetapan resmi
        tingkat kriminalitas oleh instansi terkait.

        **Batasan peta**

        Koordinat harus diperoleh dan diverifikasi dari sumber yang dapat
        dipertanggungjawabkan. Jika koordinat hanya mewakili pusat desa,
        distrik, atau alamat administratif, marker menunjukkan lokasi
        referensi tersebut, bukan lokasi persis setiap tindak kejahatan.
        """
    )
