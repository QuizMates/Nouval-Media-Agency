import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import io
import requests
import base64 # Diperlukan untuk mengkodekan gambar ke Base64
import plotly.io as pio # Diperlukan untuk mengekspor grafik Plotly sebagai gambar

# --- KONFIGURASI HALAMAN & GAYA ---
# Mengatur konfigurasi halaman. Ini harus menjadi perintah pertama Streamlit.
st.set_page_config(
    page_title="Media Intelligence Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI UTAMA & LOGIKA ---

def configure_gemini_api():
    """
    Mengkonfigurasi API Gemini menggunakan kunci API.
    Kunci API ini di-hardcode untuk tujuan demonstrasi.
    Dalam aplikasi produksi, sebaiknya gunakan st.secrets atau variabel lingkungan.
    """
    api_key = "AIzaSyC0VUu6xTFIwH3aP2R7tbhyu4O8m1ICxn4"

    if not api_key:
        st.warning("API Key Gemini tidak ditemukan. Beberapa fitur AI mungkin tidak berfungsi.")
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Gagal mengkonfigurasi Gemini API: {e}. Pastikan API Key valid.")
        return False

def get_ai_insight(prompt):
    """
    Memanggil API Gemini untuk menghasilkan wawasan berdasarkan prompt yang diberikan.
    Menggunakan model 'gemini-2.0-flash'.
    """
    # Pastikan API sudah dikonfigurasi melalui configure_gemini_api()
    if not configure_gemini_api(): # Memanggil lagi untuk memastikan konfigurasi sebelum setiap request
        return "Gagal membuat wawasan: API tidak terkonfigurasi."

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        # Mengembalikan teks dari respons. Memastikan respons memiliki struktur yang diharapkan.
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            st.error("Gemini API tidak menghasilkan teks yang valid. Respons tidak terduga.")
            return "Gagal membuat wawasan. Silakan coba lagi."
    except Exception as e:
        st.error(f"Error saat memanggil Gemini API: {e}. Pastikan API Key valid dan terhubung ke internet.")
        return "Gagal membuat wawasan: Terjadi masalah koneksi atau API."

def generate_html_report(campaign_summary, post_idea, anomaly_insight, chart_insights, chart_figures_dict, charts_to_display_info):
    """
    Membuat laporan HTML dari wawasan dan grafik yang dihasilkan AI.
    `chart_figures_dict` adalah kamus {chart_key: plotly_figure_object}.
    `charts_to_display_info` adalah daftar info grafik untuk mendapatkan judul lengkap.
    """
    current_date = pd.Timestamp.now().strftime("%d-%m-%Y %H:%M")

    anomaly_section_html = ""
    if anomaly_insight and anomaly_insight.strip() != "Belum ada wawasan yang dibuat.":
        anomaly_section_html = f"""
        <div class="section">
            <h2>3. Wawasan Anomali</h2>
            <div class="insight-box">{anomaly_insight}</div>
        </div>
        """
    
    chart_figures_html_sections = ""
    if chart_figures_dict:
        for chart_info in charts_to_display_info: # Iterate using the info to get keys and titles
            chart_key = chart_info["key"]
            chart_title = chart_info["title"]
            
            fig = chart_figures_dict.get(chart_key) # Get the figure object from the dictionary
            # Mengambil semua wawasan versi untuk chart_key
            insights_for_chart = chart_insights.get(chart_key, {}) 
            insight_text_v1 = insights_for_chart.get("gemini-2.0-flash", "Belum ada wawasan yang dibuat.")
            insight_text_v2 = insights_for_chart.get("llama-3.3-8b-instruct", "Belum ada wawasan yang dibuat.")


            if fig: # Check if a figure exists for this chart
                # Buat salinan figur untuk dimodifikasi sebelum ekspor
                fig_for_export = go.Figure(fig)
                # Atur latar belakang dan warna font untuk ekspor agar terlihat baik di laporan HTML yang terang
                fig_for_export.update_layout(
                    paper_bgcolor='#FFFFFF',  # Latar belakang kertas putih
                    plot_bgcolor='#FFFFFF',   # Latar belakang plot putih
                    font_color='#333333'      # Warna font gelap
                )
                
                try:
                    # Convert Plotly figure to PNG bytes
                    # Adjust width, height, and scale for better resolution in the report
                    img_bytes = pio.to_image(fig_for_export, format="png", width=900, height=550, scale=1.5)
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    
                    chart_figures_html_sections += f"""
                    <div class="insight-sub-section">
                        <h3>{chart_title}</h3>
                        <img src="data:image/png;base64,{img_base64}" alt="{chart_title}" style="max-width: 100%; height: auto; display: block; margin: 10px auto; border: 1px solid #ddd; border-radius: 5px;">
                        <h4>Wawasan AI Versi 1:</h4>
                        <div class="insight-box">{insight_text_v1}</div>
                        <h4>Wawasan AI Versi 2:</h4>
                        <div class="insight-box">{insight_text_v2}</div>
                    </div>
                    """
                except Exception as e:
                    chart_figures_html_sections += f"""
                    <div class="insight-sub-section">
                        <h3>{chart_title}</h3>
                        <p>Gagal menyertakan grafik ini (Error: {e}).</p>
                        <h4>Wawasan AI Versi 1:</h4>
                        <div class="insight-box">{insight_text_v1}</div>
                        <h4>Wawasan AI Versi 2:</h4>
                        <div class="insight-box">{insight_text_v2}</div>
                    </div>
                    """
            else:
                # If no figure for this specific chart_key, still include insight if available
                if insight_text_v1.strip() != "Belum ada wawasan yang dibuat." or insight_text_v2.strip() != "Belum ada wawasan yang dibuat.":
                    chart_figures_html_sections += f"""
                    <div class="insight-sub-section">
                        <h3>{chart_title}</h3>
                        <p>Tidak ada grafik yang tersedia untuk {chart_title}.</p>
                        <h4>Wawasan AI Versi 1:</h4>
                        <div class="insight-box">{insight_text_v1}</div>
                        <h4>Wawasan AI Versi 2:</h4>
                        <div class="insight-box">{insight_text_v2}</div>
                    </div>
                    """
                else:
                     chart_figures_html_sections += f"""
                    <div class="insight-sub-section">
                        <h3>{chart_title}</h3>
                        <p>Tidak ada grafik atau wawasan yang tersedia.</p>
                    </div>
                    """
    else:
        chart_figures_html_sections = "<p>Belum ada wawasan atau grafik yang dibuat.</p>"


    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Laporan Media Intelligence Dashboard</title>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: 'Inter', sans-serif; line-height: 1.6; color: #333; margin: 20px; background-color: #f4f4f4; }}
            h1, h2, h3, h4 {{ color: #2c3e50; margin-top: 1.5em; margin-bottom: 0.5em; }}
            .section {{ background-color: #fff; padding: 15px; margin-bottom: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .insight-sub-section {{ margin-top: 1em; padding-left: 15px; border-left: 3px solid #eee; }}
            .insight-box {{ background-color: #e9ecef; padding: 10px; border-radius: 5px; font-size: 0.9em; white-space: pre-wrap; word-wrap: break-word; }} /* Added word-wrap */
        </style>
    </head>
    <body>
        <h1>Laporan Media Intelligence Dashboard</h1>
        <p>Tanggal Laporan: {current_date}</p>

        <div class="section">
            <h2>1. Ringkasan Strategi Kampanye</h2>
            <div class="insight-box">{campaign_summary if campaign_summary else "Belum ada ringkasan yang dibuat."}</div>
        </div>

        <div class="section">
            <h2>2. Ide Konten AI</h2>
            <div class="insight-box">{post_idea if post_idea else "Belum ada ide postingan yang dibuat."}</div>
        </div>

        {anomaly_section_html}

        <div class="section">
            <h2>4. Wawasan Grafik</h2>
            {chart_figures_html_sections}
        </div>
        
    </body>
    </html>
    """
    return html_content.encode('utf-8') # Encoding to bytes for download

def load_css():
    """Menyuntikkan CSS kustom untuk gaya visual tingkat lanjut."""
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400..900&display=swap');

            /* Pengaturan dasar */
            body {
                background-color: #0f172a !important;
            }
            .stApp {
                background-image: radial-gradient(at top left, #1e293b, #0f172a, black);
                color: #cbd5e1;
            }
            
            /* Header */
            .main-header {
                font-family: 'Orbitron', sans-serif;
                text-align: center;
                margin-bottom: 2rem;
            }
            .main-header h1 {
                background: -webkit-linear-gradient(45deg, #06B6D4, #6366F1);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.75rem;
                font-weight: 900;
            }
            .main-header p {
                color: #94a3b8;
                font-size: 1.1rem;
            }

            /* Sidebar */
            [data-testid="stSidebar"] {
                background-color: rgba(15, 23, 42, 0.6);
                backdrop-filter: blur(10px);
                border-right: 1px solid #334155;
            }
            [data-testid="stSidebar"] h3 {
                color: #5eead4;
                font-weight: 600;
            }
            /* Menyesuaikan elemen di sidebar agar rapi */
            [data-testid="stSidebar"] .stSelectbox,
            [data-testid="stSidebar"] .stDateInput {
                margin-bottom: 1rem;
            }

            /* Container utama dan kartu */
            .stVerticalBlock > .stHorizontalBlock:nth-child(1) {
                border: 1px solid #475569;
                background-color: rgba(30, 41, 59, 0.6);
                backdrop-filter: blur(15px);
                border-radius: 1rem;
                padding: 1.5rem;
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            }
            .chart-container, .insight-hub, .anomaly-card {
                border: 1px solid #475569;
                background-color: rgba(30, 41, 59, 0.6);
                backdrop-filter: blur(15px);
                border-radius: 1rem;
                padding: 1.5rem;
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
                margin-bottom: 2rem;
                box-sizing: border-box; /* Pastikan padding dihitung dalam total lebar */
            }
             .anomaly-card {
                border: 2px solid #f59e0b;
                background-color: rgba(245, 158, 11, 0.1);
             }

            /* Kotak Wawasan AI */
            .insight-box {
                background-color: rgba(15, 23, 42, 0.7);
                border: 1px solid #334155;
                border-radius: 0.5rem;
                padding: 1rem;
                margin-top: 1rem;
                min-height: 150px;
                white-space: pre-wrap; /* Mempertahankan format teks dari AI */
                word-wrap: break-word; /* Ditambahkan: Memastikan teks panjang membungkus */
                overflow-wrap: break-word; /* Ditambahkan: Memastikan teks sangat panjang membungkus */
                font-size: 0.9rem;
            }
            
            /* Judul dalam kartu, pastikan menempel ke padding atas dan memiliki margin bawah yang konsisten */
            .chart-container h3, .insight-hub h3, .anomaly-card h3,
            .insight-hub h4 { /* Menambahkan h4 untuk judul di dalam insight-hub */
                color: #5eead4;
                margin-top: 0; /* Penting: Hapus margin atas default Streamlit */
                margin-bottom: 1rem; /* Jaga konsistensi margin bawah */
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-weight: 600;
            }

            /* Atur margin/padding untuk elemen Streamlit di dalam kotak agar lebih rapi */
            /* Menargetkan div yang dihasilkan Streamlit untuk markdown/text input */
            .anomaly-card [data-testid="stMarkdownContainer"] {
                margin: 0; /* Hapus margin default */
                padding: 0; /* Hapus padding default */
                width: 100%; /* Pastikan mengisi lebar yang tersedia */
                box-sizing: border-box; /* Penting: agar padding tidak menambah lebar total */
            }

            .anomaly-card [data-testid="stMarkdownContainer"] p {
                margin: 0; /* Hapus margin default paragraf */
                padding: 0; /* Hapus padding default paragraf */
                word-wrap: break-word;
                overflow-wrap: break-word;
                box-sizing: border-box;
            }

            /* Menargetkan teks di dalam div anomali card secara langsung */
            /* Ini penting karena st.markdown di dalam div anomaly-text di masa lalu */
            /* Sekarang, jika langsung st.markdown, atur marginnya */
            .anomaly-card p { /* Target p tags directly inside anomaly-card (if not wrapped by stMarkdownContainer) */
                margin-top: 0.5rem;
                margin-bottom: 1rem;
                padding: 0; /* Hapus padding horizontal dari anomaly-text */
                word-wrap: break-word;
                overflow-wrap: break-word;
            }
            
            /* Menargetkan elemen <p> yang dihasilkan oleh st.write atau st.markdown biasa di luar anomaly-card, di dalam chart-container/insight-hub */
            .chart-container .stMarkdown p, 
            .insight-hub .stMarkdown p {
                margin-top: 0.25rem; /* Sesuaikan margin atas */
                margin-bottom: 0.5rem; /* Sesuaikan margin bawah */
                word-wrap: break-word; /* Ditambahkan: Memastikan teks panjang membungkus */
                overflow-wrap: break-word; /* Ditambahkan: Memastikan teks sangat panjang membungkus */
            }

            .chart-container > div > div > div > .stFileUploader {
                margin-bottom: 1rem;
            }

            .chart-container > div > div > div > .stButton,
            .insight-hub > div > div > div > .stButton,
            .anomaly-card > div > div > div > .stButton {
                margin-bottom: 0.5rem;
                margin-top: 1rem; /* Tambahkan sedikit ruang di atas tombol */
            }
            /* Styling untuk tombol radio wawasan */
            div[data-testid="stRadio"] label {
                margin-bottom: 0.5rem; /* Beri jarak antar opsi radio */
            }

            /* Gaya untuk menampilkan info file terunggah */
            .uploaded-file-info {
                background-color: rgba(30, 41, 59, 0.6);
                border: 1px solid #475569;
                border-radius: 1rem;
                padding: 1.5rem;
                margin-bottom: 2rem;
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
                color: #cbd5e1;
            }
            .uploaded-file-info h3 {
                color: #5eead4;
                margin-top: 0;
                margin-bottom: 1rem;
            }
            .uploaded-file-info p {
                margin-bottom: 0.5rem;
            }
            .uploaded-file-info .stButton {
                margin-top: 1.5rem;
            }
        </style>
    """, unsafe_allow_html=True)

@st.cache_data
def parse_csv(uploaded_file):
    """Membaca file CSV yang diunggah ke dalam DataFrame pandas dan membersihkannya."""
    try:
        string_data = uploaded_file.getvalue().decode("utf-8")
        df = pd.read_csv(io.StringIO(string_data))
        
        # --- PERBAIKAN: Mengganti nama kolom 'Media_Type' menjadi 'Media Type' ---
        if 'Media_Type' in df.columns:
            df.rename(columns={'Media_Type': 'Media Type'}, inplace=True)
        # ----------------------------------------------------------------------
        
        # Pembersihan data
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Engagements'] = pd.to_numeric(df['Engagements'], errors='coerce')
        df.dropna(subset=['Date', 'Engagements'], inplace=True)
        df['Engagements'] = df['Engagements'].astype(int)
        
        # Memastikan kolom lain ada
        required_cols = ['Platform', 'Sentiment', 'Media Type', 'Location', 'Headline']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 'N/A' # Isi dengan N/A jika tidak ada
        df[required_cols] = df[required_cols].fillna('N/A')

        return df
    except Exception as e:
        st.error(f"Gagal memproses file CSV. Pastikan formatnya benar. Error: {e}")
        return None

# --- UI STREAMLIT ---
load_css()
api_configured = configure_gemini_api() # Panggil fungsi konfigurasi API di awal

# Header Utama
st.markdown("""
    <div class="main-header">
        <h1>Media Intelligence Dashboard</h1>
        <p>Ryan Vandiaz Media Agency</p>
    </div>
""", unsafe_allow_html=True)


# Inisialisasi session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'chart_insights' not in st.session_state:
    st.session_state.chart_insights = {} # Sekarang akan menyimpan dict {"key": {"Versi 1": ..., "Versi 2": ...}}
if 'campaign_summary' not in st.session_state:
    st.session_state.campaign_summary = ""
if 'post_idea' not in st.session_state:
    st.session_state.post_idea = ""
if 'anomaly_insight' not in st.session_state:
    st.session_state.anomaly_insight = ""
if 'chart_figures' not in st.session_state:
    st.session_state.chart_figures = {}
if 'last_uploaded_file_name' not in st.session_state: # Tambahkan inisialisasi untuk nama file
    st.session_state.last_uploaded_file_name = None
if 'last_uploaded_file_size' not in st.session_state: # Tambahkan inisialisasi untuk ukuran file
    st.session_state.last_uploaded_file_size = None


# Tampilan unggah file (hanya muncul jika data belum diunggah)
if st.session_state.data is None: 
    with st.container():
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("### ☁️ Unggah File CSV Anda")
            st.write("Pastikan file memiliki kolom 'Date', 'Engagements', 'Sentiment', 'Platform', 'Media_Type', 'Location', dan (opsional) 'Headline'.")
            uploaded_file = st.file_uploader("Pilih file CSV", type="csv", key="main_file_uploader")
            if uploaded_file is not None:
                # Periksa apakah file yang diunggah sama dengan yang terakhir kali diproses
                # Ini penting untuk menghindari pemrosesan ulang saat widget file_uploader mempertahankan nilainya
                if uploaded_file.name != st.session_state.last_uploaded_file_name or uploaded_file.size != st.session_state.last_uploaded_file_size:
                    st.session_state.data = parse_csv(uploaded_file)
                    if st.session_state.data is not None:
                        # Simpan detail file
                        st.session_state.last_uploaded_file_name = uploaded_file.name
                        st.session_state.last_uploaded_file_size = uploaded_file.size
                        st.rerun() # PERBAIKAN: Memaksa rerun untuk menyembunyikan bagian unggah dan menampilkan dashboard
                # else:
                #     st.info("File ini sudah diunggah dan dianalisis.") # Opsional: pesan jika file yang sama diunggah ulang
            st.markdown('</div>', unsafe_allow_html=True)


# Tampilan Dasbor Utama (setelah file diunggah)
if st.session_state.data is not None:
    df = st.session_state.data

    # Menampilkan informasi file yang terunggah dan tombol hapus
    st.markdown(f"""
        <div class="uploaded-file-info">
            <h3>☁️ File Terunggah</h3>
            <p><strong>Nama File:</strong> {st.session_state.last_uploaded_file_name}</p>
            <p><strong>Ukuran File:</strong> {st.session_state.last_uploaded_file_size / (1024 * 1024):.2f} MB</p>
            <p style="color: #5eead4; font-weight: bold;">File CSV berhasil diunggah dan diproses!</p>
    """, unsafe_allow_html=True)
    
    if st.button("Hapus File Terunggah", key="clear_file_btn"):
        st.session_state.data = None # Hapus data
        st.session_state.chart_insights = {} # Bersihkan wawasan
        st.session_state.campaign_summary = ""
        st.session_state.post_idea = ""
        st.session_state.anomaly_insight = ""
        st.session_state.chart_figures = {}
        st.session_state.last_filter_state = "" # Reset filter state
        st.session_state.last_uploaded_file_name = None # Hapus info file
        st.session_state.last_uploaded_file_size = None # Hapus info file
        st.rerun() # Rerun aplikasi untuk menampilkan kembali bagian upload
    st.markdown('</div>', unsafe_allow_html=True) # Tutup div uploaded-file-info


    # Notifikasi "Berikut adalah hasil analisis datamu."
    st.info("Berikut adalah hasil analisis datamu.")


    # --- Sidebar Filter ---
    with st.sidebar:
        st.markdown("<h3><i class='fas fa-filter'></i> Filter Data</h3>", unsafe_allow_html=True)

        platform = st.selectbox("Platform", ["All"] + list(df['Platform'].unique()), key='platform_filter')
        sentiment = st.selectbox("Sentiment", ["All"] + list(df['Sentiment'].unique()), key='sentiment_filter')
        media_type = st.selectbox("Media Type", ["All"] + list(df['Media Type'].unique()), key='media_type_filter')
        location = st.selectbox("Location", ["All"] + list(df['Location'].unique()), key='location_filter')

        min_date, max_date = df['Date'].min().date(), df['Date'].max().date()
        start_date = st.date_input("Tanggal Mulai", min_date, min_value=min_date, max_value=max_date, key='start_date_filter')
        end_date = st.date_input("Tanggal Akhir", max_date, min_value=min_date, max_value=max_date, key='end_date_filter')
        
        # Logika reset insight jika filter berubah
        filter_state = f"{platform}{sentiment}{media_type}{location}{start_date}{end_date}"
        if 'last_filter_state' not in st.session_state or st.session_state.last_filter_state != filter_state:
            st.session_state.chart_insights = {}
            st.session_state.campaign_summary = ""
            st.session_state.post_idea = ""
            st.session_state.anomaly_insight = ""
            st.session_state.chart_figures = {} # Reset chart figures juga
            st.session_state.last_filter_state = filter_state


    # Filter DataFrame
    filtered_df = df[
        (df['Date'].dt.date >= start_date) &
        (df['Date'].dt.date <= end_date)
    ]
    if platform != "All":
        filtered_df = filtered_df[filtered_df['Platform'] == platform]
    if sentiment != "All":
        filtered_df = filtered_df[filtered_df['Sentiment'] == sentiment]
    if media_type != "All":
        filtered_df = filtered_df[filtered_df['Media Type'] == media_type]
    if location != "All":
        filtered_df = filtered_df[filtered_df['Location'] == location]

    # --- Pusat Wawasan AI ---
    st.markdown('<div class="insight-hub">', unsafe_allow_html=True)
    st.markdown("<h3>🧠 Pusat Wawasan AI</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h4>⚡ Ringkasan Strategi Kampanye</h4>", unsafe_allow_html=True)
        if st.button("Buat Ringkasan", key="summary_btn", use_container_width=True):
            with st.spinner("Membuat ringkasan strategi..."):
                prompt = f"""
                Anda adalah seorang konsultan strategi media senior. Analisis data kampanye berikut secara komprehensif. Berikan ringkasan eksekutif (3-4 kalimat) diikuti oleh 3 rekomendasi strategis utama yang paling berdampak. 
                Gunakan data berikut:
                - Data yang difilter (5 baris pertama): {filtered_df.head().to_json()}
                - Jumlah total sebutan: {len(filtered_df)}
                - Rata-rata keterlibatan: {filtered_df['Engagements'].mean():.2f}
                - Distribusi Sentimen: {filtered_df['Sentiment'].value_counts().to_json()}
                - Keterlibatan per Platform: {filtered_df.groupby('Platform')['Engagements'].sum().to_json()}
                Fokus pada gambaran besar: Apa cerita utama yang disampaikan oleh data ini? Di mana peluang terbesar dan apa risiko utamanya? Format jawaban Anda dengan jelas.
                """
                summary = get_ai_insight(prompt)
                st.session_state.campaign_summary = summary
                
        st.markdown(f'<div class="insight-box">{st.session_state.campaign_summary or "Klik untuk membuat ringkasan strategis dari semua data."}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("<h4>💡 Generator Ide Konten AI</h4>", unsafe_allow_html=True)
        if st.button("✨ Buat Ide Postingan", key="idea_btn", use_container_width=True):
            with st.spinner("Mencari ide terbaik..."):
                if not filtered_df.empty:
                    best_platform_series = filtered_df.groupby('Platform')['Engagements'].sum()
                    if not best_platform_series.empty:
                        best_platform = best_platform_series.idxmax()
                        top_posts = filtered_df[filtered_df['Platform'] == best_platform].nlargest(5, 'Engagements')
                        top_headlines = ', '.join(top_posts['Headline'].tolist())
                    else:
                        best_platform = "tidak diketahui"
                        top_headlines = "tidak ada data"
                else:
                    best_platform = "tidak diketahui"
                    top_headlines = "tidak ada data"


                prompt = f"""
                Anda adalah seorang ahli strategi media sosial yang kreatif. Berdasarkan data berikut, buatlah satu contoh postingan untuk platform **{best_platform}**.
                - Platform Berkinerja Terbaik: {best_platform}
                - Topik Berkinerja Tinggi (dari judul): {top_headlines}
                Postingan harus:
                1. Ditulis dalam Bahasa Indonesia.
                2. Memiliki nada yang menarik dan sesuai untuk {best_platform}.
                3. Memberikan saran konsep visual yang jelas.
                4. Menyertakan 3-5 tagar yang relevan.
                Format output dengan jelas: "Platform:", "Konten Postingan:", "Saran Visual:", dan "Tagar:".
                """
                idea = get_ai_insight(prompt)
                st.session_state.post_idea = idea
                
        st.markdown(f'<div class="insight-box">{st.session_state.post_idea or "Klik untuk menghasilkan ide postingan berdasarkan data kinerja terbaik Anda."}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Deteksi Anomali ---
    engagement_trend = filtered_df.groupby(filtered_df['Date'].dt.date)['Engagements'].sum().reset_index()
    if len(engagement_trend) > 7:
        mean = engagement_trend['Engagements'].mean()
        std = engagement_trend['Engagements'].std()
        anomaly_threshold = mean + (2 * std) if std > 0 else mean + 0.1
        anomalies = engagement_trend[engagement_trend['Engagements'] > anomaly_threshold]
        
        if not anomalies.empty:
            anomaly = anomalies.iloc[0]
            st.markdown('<div class="anomaly-card">', unsafe_allow_html=True)
            st.markdown("<h3>⚠️ Peringatan Anomali Terdeteksi!</h3>", unsafe_allow_html=True)
            # Menggunakan st.markdown langsung (tanpa div anomaly-text) untuk pesan di dalam card
            st.markdown(f"""
                Kami mendeteksi lonjakan keterlibatan yang tidak biasa pada **{anomaly['Date']}** dengan **{int(anomaly['Engagements']):,}** keterlibatan (rata-rata: {int(mean):,} &plusmn; {int(std):,}).
            """, unsafe_allow_html=True)
            
            if st.button("✨ Jelaskan Anomali Ini", key="anomaly_btn"):
                with st.spinner("Menganalisis penyebab anomali..."):
                    anomaly_day_data = filtered_df[filtered_df['Date'].dt.date == anomaly['Date']]
                    top_headlines_on_anomaly_day = ', '.join(anomaly_day_data.nlargest(3, 'Engagements')['Headline'].tolist())

                    prompt = f"""
                    Anda adalah seorang analis data intelijen media. Terdeteksi anomali keterlibatan pada {anomaly['Date']}.
                    - Keterlibatan pada hari itu: {anomaly['Engagements']}
                    - Rata-rata keterlibatan historis (dari data yang difilter): {mean:.2f}
                    - Judul berita teratas yang berkontribusi pada hari itu (jika ada): {top_headlines_on_anomaly_day if top_headlines_on_anomaly_day else "Tidak ada judul spesifik yang ditemukan."}
                    Berikan 3 kemungkinan penyebab anomali ini dan 2 rekomendasi tindakan. Format sebagai daftar bernomor.
                    """
                    insight = get_ai_insight(prompt)
                    st.session_state.anomaly_insight = insight

            if st.session_state.anomaly_insight:
                st.markdown(f'<div class="insight-box">{st.session_state.anomaly_insight}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


    # --- Tampilan Grafik ---
    chart_cols = st.columns(2)
    
    # Daftar grafik untuk ditampilkan.
    charts_to_display = [
        {"key": "sentiment", "title": "Analisis Sentimen", "col": chart_cols[0]},
        {"key": "trend", "title": "Tren Keterlibatan Seiring Waktu", "col": chart_cols[1]},
        {"key": "platform", "title": "Keterlibatan per Platform", "col": chart_cols[0]},
        {"key": "mediaType", "title": "Distribusi Jenis Media", "col": chart_cols[1]}, 
        {"key": "location", "title": "5 Lokasi Teratas", "col": chart_cols[0]},
    ]
    
    # Fungsi pembuat prompt untuk menghindari pengulangan
    def get_chart_prompt(key, data_json, version_type="Versi 1"):
        base_prompts = {
            "sentiment": f"Berdasarkan data distribusi sentimen berikut: {data_json}, berikan 3 wawasan (insights) tajam dan dapat ditindaklanjuti untuk strategi komunikasi merek. Format sebagai daftar bernomor.",
            "trend": f"Berdasarkan 10 data tren keterlibatan harian terakhir ini: {data_json}, berikan 3 wawasan strategis tentang puncak, penurunan, dan pola umum. Apa artinya ini bagi ritme kampanye? Format sebagai daftar bernomor.",
            "platform": f"Berdasarkan data keterlibatan per platform ini: {data_json}, berikan 3 wawasan yang dapat ditindaklanjuti. Identifikasi platform 'juara' dan 'peluang'. Format sebagai daftar bernomor.",
            "mediaType": f"Berdasarkan data distribusi jenis media ini: {data_json}, berikan 3 wawasan strategis. Analisis preferensi audiens berdasarkan format konten. Format sebagai daftar bernomor.",
            "location": f"Berdasarkan data keterlibatan per lokasi ini: {data_json}, berikan 3 wawasan geo-strategis. Identifikasi pasar utama dan pasar yang sedang berkembang. Format sebagai daftar bernomor."
        }
        
        prompt_modifier = ""
        if version_type == "Versi 2":
            prompt_modifier = "Dari perspektif yang berbeda, atau fokus pada peluang tersembunyi/risiko yang diabaikan, berikan 3 wawasan alternatif/pelengkap. Hindari pengulangan ide dari wawasan umum."
            
        return f"Anda adalah seorang konsultan intelijen media profesional. {base_prompts.get(key, '')} {prompt_modifier}".strip()


    for chart in charts_to_display:
        with chart["col"]:
            st.markdown(f'<div class="chart-container" key="chart-{chart["key"]}">', unsafe_allow_html=True)
            st.markdown(f'<h3>{chart["title"]}</h3>', unsafe_allow_html=True)
            
            fig = None
            chart_data_for_prompt = None

            if chart["key"] == "sentiment":
                sentiment_data = filtered_df['Sentiment'].value_counts().reset_index()
                sentiment_data.columns = ['Sentiment', 'count']
                if not sentiment_data.empty:
                    fig = px.pie(sentiment_data, names='Sentiment', values='count', color_discrete_sequence=px.colors.qualitative.Pastel)
                chart_data_for_prompt = sentiment_data.to_json()

            elif chart["key"] == "trend":
                engagement_trend_chart = filtered_df.groupby(filtered_df['Date'].dt.date)['Engagements'].sum().reset_index()
                engagement_trend_chart['Date'] = pd.to_datetime(engagement_trend_chart['Date'])
                if not engagement_trend_chart.empty:
                    fig = px.line(engagement_trend_chart, x='Date', y='Engagements', labels={'Date': 'Tanggal', 'Engagements': 'Total Keterlibatan'})
                    fig.update_traces(line=dict(color='#06B6D4', width=3))
                chart_data_for_prompt = engagement_trend_chart.tail(10).to_json()
                
            elif chart["key"] == "platform":
                platform_data = filtered_df.groupby('Platform')['Engagements'].sum().sort_values(ascending=False).reset_index()
                if not platform_data.empty:
                    fig = px.bar(platform_data, x='Platform', y='Engagements', color='Platform')
                chart_data_for_prompt = platform_data.to_json()

            elif chart["key"] == "mediaType":
                media_type_data = filtered_df['Media Type'].value_counts().reset_index()
                media_type_data.columns = ['Media Type', 'count']
                if not media_type_data.empty:
                    fig = px.pie(media_type_data, names='Media Type', values='count', hole=.3,
                                 color_discrete_map={
                                     'Image': '#6366F1',
                                     'Video': '#06B6D4',
                                     'Text': '#5EEAD4',
                                     'Carousel': '#F59E0B'
                                 })
                chart_data_for_prompt = media_type_data.to_json()

            elif chart["key"] == "location": 
                location_data = filtered_df.groupby('Location')['Engagements'].sum().nlargest(5).reset_index()
                if not location_data.empty:
                    fig = px.bar(location_data, y='Location', x='Engagements', orientation='h', color='Location')
                chart_data_for_prompt = location_data.to_json()
            
            if fig: # Hanya tampilkan grafik jika ada data untuk dibuat
                st.session_state.chart_figures[chart["key"]] = fig 

                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#cbd5e1',
                    legend_title_text=''
                )
                st.plotly_chart(fig, use_container_width=True)

                # Tombol untuk membuat wawasan AI untuk setiap grafik
                if st.button(f"✨ Generate AI Insight  ({chart['title']})", key=f"insight_btn_{chart['key']}"):
                    with st.spinner(f"Menganalisis {chart['title']} dan membuat wawasan..."):
                        if chart_data_for_prompt:
                            prompt_v1 = get_chart_prompt(chart['key'], chart_data_for_prompt, "Versi 1")
                            insight_v1 = get_ai_insight(prompt_v1)
                            
                            prompt_v2 = get_chart_prompt(chart['key'], chart_data_for_prompt, "Versi 2")
                            insight_v2 = get_ai_insight(prompt_v2)
                            
                            st.session_state.chart_insights[chart['key']] = {
                                "gemini-2.0-flash": insight_v1,
                                "llama-3.3-8b-instruct": insight_v2
                            }
                        else:
                            st.session_state.chart_insights[chart['key']] = {
                                "gemini-2.0-flash": "Tidak ada data yang cukup untuk menghasilkan wawasan.",
                                "llama-3.3-8b-instruct": "Tidak ada data yang cukup untuk menghasilkan wawasan."
                            }
                
                # Selector untuk versi wawasan
                current_insights = st.session_state.chart_insights.get(chart['key'], {})
                
                # Tambahkan teks info di atas tombol pilih versi
                st.markdown("Pilih Model Ai  AI untuk ditampilkan:", unsafe_allow_html=True)
                selected_insight_version = st.radio(
                    "Pilih Model Ai :",
                    ("gemini-2.0-flash", "llama-3.3-8b-instruct"),
                    key=f"insight_selector_{chart['key']}"
                )
                
                insight_text_to_display = current_insights.get(selected_insight_version, "Klik 'Buat Wawasan AI' untuk menghasilkan wawasan.")
                
                st.markdown(f'<div class="insight-box">{insight_text_to_display}</div>', unsafe_allow_html=True)
            else:
                st.write("Tidak ada data yang tersedia untuk grafik ini dengan filter yang dipilih.")
                st.session_state.chart_figures[chart["key"]] = None 

                if st.button(f"✨ Generate AI Insight  ({chart['title']})", key=f"insight_btn_{chart['key']}_no_chart"):
                    st.session_state.chart_insights[chart['key']] = {
                        "gemini-2.0-flash": "Tidak ada data yang cukup untuk menghasilkan wawasan.",
                        "llama-3.3-8b-instruct": "Tidak ada data yang cukup untuk menghasilkan wawasan."
                    }
                
                # Selector untuk versi wawasan (bahkan jika tidak ada grafik)
                # Tambahkan teks info di atas tombol pilih versi
                st.markdown("Pilih Model Ai  AI untuk ditampilkan:", unsafe_allow_html=True)
                current_insights = st.session_state.chart_insights.get(chart['key'], {})
                selected_insight_version = st.radio(
                    "Pilih Model Ai :",
                    ("gemini-2.0-flash", "llama-3.3-8b-instruct"),
                    key=f"insight_selector_{chart['key']}_no_chart"
                )
                insight_text_to_display = current_insights.get(selected_insight_version, "Klik 'Buat Wawasan AI' untuk menghasilkan wawasan.")
                st.markdown(f'<div class="insight-box">{insight_text_to_display}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Bagian Unduh Laporan HTML ---
    st.markdown("---")
    st.markdown("<h3>📄 Unduh Laporan Analisis</h3>", unsafe_allow_html=True)
    
    # Kumpulkan semua wawasan dan objek grafik untuk laporan HTML
    chart_insights_for_report = {
        chart_info["key"]: st.session_state.chart_insights.get(chart_info["key"], {}) 
        for chart_info in charts_to_display
    }

    if st.button("Unduh Laporan HTML", key="download_html_btn", type="primary", use_container_width=True):
        with st.spinner("Membangun laporan HTML dengan grafik..."):
            html_data = generate_html_report(
                st.session_state.campaign_summary,
                st.session_state.post_idea,
                st.session_state.anomaly_insight,
                chart_insights_for_report, 
                st.session_state.chart_figures,
                charts_to_display
            )
            
            if html_data:
                st.download_button(
                    label="Klik untuk Mengunduh",
                    data=html_data,
                    file_name="Laporan_Media_Intelligence.html",
                    mime="text/html",
                    key="actual_download_button_html"
                )
                st.success("Laporan HTML siap diunduh! Buka file ini di browser Anda, lalu gunakan fitur cetak browser untuk menyimpannya sebagai PDF jika diperlukan.")
            else:
                st.error("Gagal membuat laporan HTML. Pastikan semua grafik telah dibuat atau ada data.")
