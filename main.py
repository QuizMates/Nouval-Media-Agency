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
    if not configure_gemini_api():
        return "Gagal membuat wawasan: API tidak terkonfigurasi."

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
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
        for chart_info in charts_to_display_info:
            chart_key = chart_info["key"]
            chart_title = chart_info["title"]
            
            fig = chart_figures_dict.get(chart_key)
            insights_for_chart = chart_insights.get(chart_key, {}) 
            insight_text_v1 = insights_for_chart.get("gemini-2.0-flash", "Belum ada wawasan yang dibuat.")
            insight_text_v2 = insights_for_chart.get("llama-3.3-8b-instruct", "Belum ada wawasan yang dibuat.")

            if fig:
                fig_for_export = go.Figure(fig)
                fig_for_export.update_layout(
                    paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF', font_color='#333333'
                )
                
                try:
                    img_bytes = pio.to_image(fig_for_export, format="png", width=900, height=550, scale=1.5)
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    
                    chart_figures_html_sections += f"""
                    <div class="insight-sub-section">
                        <h3>{chart_title}</h3>
                        <img src="data:image/png;base64,{img_base64}" alt="{chart_title}" style="max-width: 100%; height: auto; display: block; margin: 10px auto; border: 1px solid #ddd; border-radius: 8px;">
                        <h4>Wawasan dari gemini-2.0-flash:</h4>
                        <div class="insight-box">{insight_text_v1}</div>
                        <h4>Wawasan dari llama-3.3-8b-instruct:</h4>
                        <div class="insight-box">{insight_text_v2}</div>
                    </div>
                    """
                except Exception as e:
                    chart_figures_html_sections += f"<p>Gagal menyertakan grafik: {chart_title} (Error: {e})</p>"
            else:
                if insight_text_v1.strip() != "Belum ada wawasan yang dibuat." or insight_text_v2.strip() != "Belum ada wawasan yang dibuat.":
                    chart_figures_html_sections += f"""
                    <div class="insight-sub-section">
                        <h3>{chart_title}</h3><p>Grafik tidak tersedia.</p>
                        <h4>Wawasan dari gemini-2.0-flash:</h4><div class="insight-box">{insight_text_v1}</div>
                        <h4>Wawasan dari llama-3.3-8b-instruct:</h4><div class="insight-box">{insight_text_v2}</div>
                    </div>"""
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
            h1, h2, h3, h4 {{ color: #2c3e50; }}
            .section {{ background-color: #fff; padding: 15px; margin-bottom: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .insight-sub-section {{ margin-top: 1em; padding-left: 15px; border-left: 3px solid #eee; }}
            .insight-box {{ background-color: #e9ecef; padding: 10px; border-radius: 5px; font-size: 0.9em; white-space: pre-wrap; word-wrap: break-word; }}
        </style>
    </head>
    <body>
        <h1>Laporan Media Intelligence Dashboard</h1>
        <p>Tanggal Laporan: {current_date}</p>
        <div class="section"><h2>1. Ringkasan Strategi Kampanye</h2><div class="insight-box">{campaign_summary or "Belum ada ringkasan."}</div></div>
        <div class="section"><h2>2. Ide Konten AI</h2><div class="insight-box">{post_idea or "Belum ada ide."}</div></div>
        {anomaly_section_html}
        <div class="section"><h2>4. Wawasan Grafik</h2>{chart_figures_html_sections}</div>
    </body>
    </html>
    """
    return html_content.encode('utf-8')

def load_css():
    """Menyuntikkan CSS kustom untuk gaya visual modern dengan gradien."""
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

            :root {
                --bg-color-start: #1a1c2c;
                --bg-color-end: #0a0a14;
                --primary-text-color: #f0f2f5;
                --secondary-text-color: #a1a1aa;
                --card-bg-color: rgba(255, 255, 255, 0.05);
                --card-border-color: rgba(255, 255, 255, 0.1);
                --accent-gradient: linear-gradient(90deg, #00f2fe, #4facfe);
                --accent-color-1: #00f2fe;
                --accent-color-2: #4facfe;
                --red-accent: #ff5252;
            }

            body {
                background-color: var(--bg-color-start) !important;
                font-family: 'Inter', sans-serif;
            }
            .stApp {
                background-image: linear-gradient(135deg, var(--bg-color-start) 0%, var(--bg-color-end) 100%);
                color: var(--primary-text-color);
            }
            
            .main-header {
                text-align: center;
                margin-bottom: 3rem;
                padding-top: 2rem;
            }
            .main-header h1 {
                background: var(--accent-gradient);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 3.25rem;
                font-weight: 800;
            }
            .main-header p {
                color: var(--secondary-text-color);
                font-size: 1.25rem;
                margin-top: 0.25rem;
            }

            [data-testid="stSidebar"] {
                background-color: rgba(10, 10, 20, 0.8);
                backdrop-filter: blur(10px);
                border-right: 1px solid var(--card-border-color);
            }
            [data-testid="stSidebar"] h3 {
                color: var(--primary-text-color);
                font-weight: 600;
            }

            .chart-container, .insight-hub, .anomaly-card, .uploaded-file-info {
                background: var(--card-bg-color);
                backdrop-filter: blur(5px);
                border: 1px solid var(--card-border-color);
                border-radius: 1rem;
                padding: 1.75rem;
                margin-bottom: 2rem;
                box-sizing: border-box;
            }
             .anomaly-card {
                border-left: 4px solid var(--red-accent);
             }

            .chart-container h3, .insight-hub h3, .anomaly-card h3, .uploaded-file-info h3,
            .insight-hub h4 {
                color: var(--primary-text-color);
                margin-top: 0;
                margin-bottom: 1rem;
                font-weight: 600;
                font-size: 1.2rem;
            }
            .insight-hub h4 {
                 color: var(--secondary-text-color);
                 font-size: 1rem;
                 font-weight: 500;
            }

            .insight-box {
                background-color: rgba(0,0,0,0.2);
                border: 1px solid var(--card-border-color);
                border-radius: 0.75rem;
                padding: 1.25rem;
                margin-top: 1rem;
                min-height: 150px;
                font-size: 0.95rem;
                color: var(--secondary-text-color);
                white-space: pre-wrap;
                word-wrap: break-word;
            }

            .stButton > button {
                background: var(--card-bg-color);
                color: var(--primary-text-color);
                border: 1px solid var(--card-border-color);
                border-radius: 0.5rem;
                padding: 0.7rem 1.4rem;
                font-weight: 600;
                transition: all 0.2s ease-in-out;
            }
            .stButton > button:hover {
                border-color: var(--accent-color-1);
                color: var(--accent-color-1);
            }
            
            .stButton > button[kind="primary"] {
                background: var(--accent-gradient);
                color: white;
                font-weight: 700;
                border: none;
            }
            .stButton > button[kind="primary"]:hover {
                 filter: brightness(1.2);
                 box-shadow: 0 0 15px rgba(0, 242, 254, 0.4);
            }

            div[data-testid="stRadio"] label { padding: 0.5rem 0; }
            .stFileUploader, .stDateInput, .stSelectbox { margin-bottom: 1rem; }
        </style>
    """, unsafe_allow_html=True)

@st.cache_data
def parse_csv(uploaded_file):
    """Membaca file CSV yang diunggah ke dalam DataFrame pandas dan membersihkannya."""
    try:
        string_data = uploaded_file.getvalue().decode("utf-8")
        df = pd.read_csv(io.StringIO(string_data))
        
        if 'Media_Type' in df.columns:
            df.rename(columns={'Media_Type': 'Media Type'}, inplace=True)
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Engagements'] = pd.to_numeric(df['Engagements'], errors='coerce')
        df.dropna(subset=['Date', 'Engagements'], inplace=True)
        df['Engagements'] = df['Engagements'].astype(int)
        
        required_cols = ['Platform', 'Sentiment', 'Media Type', 'Location', 'Headline']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 'N/A'
        df[required_cols] = df[required_cols].fillna('N/A')

        return df
    except Exception as e:
        st.error(f"Gagal memproses file CSV. Pastikan formatnya benar. Error: {e}")
        return None

# --- UI STREAMLIT ---
load_css()
api_configured = configure_gemini_api()

st.markdown("""
    <div class="main-header">
        <h1>Media Intelligence Dashboard</h1>
        <p>Ryan Vandiaz Media Agency</p>
    </div>
""", unsafe_allow_html=True)

# Inisialisasi session state
if 'data' not in st.session_state: st.session_state.data = None
if 'chart_insights' not in st.session_state: st.session_state.chart_insights = {}
if 'campaign_summary' not in st.session_state: st.session_state.campaign_summary = ""
if 'post_idea' not in st.session_state: st.session_state.post_idea = ""
if 'anomaly_insight' not in st.session_state: st.session_state.anomaly_insight = ""
if 'chart_figures' not in st.session_state: st.session_state.chart_figures = {}
if 'last_uploaded_file_name' not in st.session_state: st.session_state.last_uploaded_file_name = None
if 'last_uploaded_file_size' not in st.session_state: st.session_state.last_uploaded_file_size = None

# Tampilan unggah file
if st.session_state.data is None: 
    with st.container():
        _, col2, _ = st.columns([1,2,1])
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("### ☁️ Unggah File CSV Anda")
            st.write("Mulai dengan mengunggah data media Anda untuk dianalisis.")
            uploaded_file = st.file_uploader("Pilih file CSV", type="csv", key="main_file_uploader", label_visibility="collapsed")
            if uploaded_file:
                st.session_state.data = parse_csv(uploaded_file)
                if st.session_state.data is not None:
                    st.session_state.last_uploaded_file_name = uploaded_file.name
                    st.session_state.last_uploaded_file_size = uploaded_file.size
                    st.rerun() 
            st.markdown('</div>', unsafe_allow_html=True)

# Tampilan Dasbor Utama
if st.session_state.data is not None:
    df = st.session_state.data

    st.markdown('<div class="uploaded-file-info">', unsafe_allow_html=True)
    st.markdown(f"""
        <h3>☁️ File Terunggah: {st.session_state.last_uploaded_file_name}</h3>
        <p>Ukuran: {st.session_state.last_uploaded_file_size / 1024:.2f} KB</p>
    """, unsafe_allow_html=True)
    
    if st.button("Hapus File & Reset", key="clear_file_btn"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar Filter
    with st.sidebar:
        st.markdown("<h3>Filter Data</h3>", unsafe_allow_html=True)
        platform = st.selectbox("Platform", ["All"] + list(df['Platform'].unique()), key='platform_filter')
        sentiment = st.selectbox("Sentiment", ["All"] + list(df['Sentiment'].unique()), key='sentiment_filter')
        media_type = st.selectbox("Media Type", ["All"] + list(df['Media Type'].unique()), key='media_type_filter')
        location = st.selectbox("Location", ["All"] + list(df['Location'].unique()), key='location_filter')
        start_date, end_date = st.date_input("Rentang Tanggal", [df['Date'].min().date(), df['Date'].max().date()], key='date_range_filter')
        
        filter_state = f"{platform}{sentiment}{media_type}{location}{start_date}{end_date}"
        if 'last_filter_state' not in st.session_state or st.session_state.last_filter_state != filter_state:
            st.session_state.chart_insights = {}
            st.session_state.campaign_summary, st.session_state.post_idea, st.session_state.anomaly_insight = "", "", ""
            st.session_state.chart_figures = {}
            st.session_state.last_filter_state = filter_state

    filtered_df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
    if platform != "All": filtered_df = filtered_df[filtered_df['Platform'] == platform]
    if sentiment != "All": filtered_df = filtered_df[filtered_df['Sentiment'] == sentiment]
    if media_type != "All": filtered_df = filtered_df[filtered_df['Media Type'] == media_type]
    if location != "All": filtered_df = filtered_df[filtered_df['Location'] == location]

    # Pusat Wawasan AI
    st.markdown('<div class="insight-hub">', unsafe_allow_html=True)
    st.markdown("<h3>🧠 Pusat Wawasan AI</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4>Ringkasan Strategi Kampanye</h4>", unsafe_allow_html=True)
        if st.button("Buat Ringkasan", key="summary_btn", use_container_width=True):
            with st.spinner("Membuat ringkasan..."):
                prompt = f"Anda adalah konsultan strategi. Analisis data ini: {filtered_df.describe().to_json()}. Berikan ringkasan eksekutif dan 3 rekomendasi strategis."
                st.session_state.campaign_summary = get_ai_insight(prompt)
        st.markdown(f'<div class="insight-box">{st.session_state.campaign_summary or "Klik untuk membuat ringkasan strategis."}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("<h4>Generator Ide Konten AI</h4>", unsafe_allow_html=True)
        if st.button("Buat Ide Postingan", key="idea_btn", use_container_width=True):
            with st.spinner("Mencari ide..."):
                best_platform = filtered_df.groupby('Platform')['Engagements'].sum().idxmax()
                prompt = f"Anda adalah ahli media sosial. Buat satu contoh postingan untuk platform {best_platform}, termasuk saran visual dan 3-5 tagar."
                st.session_state.post_idea = get_ai_insight(prompt)
        st.markdown(f'<div class="insight-box">{st.session_state.post_idea or "Klik untuk menghasilkan ide postingan."}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Deteksi Anomali
    engagement_trend = filtered_df.groupby(filtered_df['Date'].dt.date)['Engagements'].sum().reset_index()
    if len(engagement_trend) > 7:
        mean = engagement_trend['Engagements'].mean()
        std = engagement_trend['Engagements'].std()
        anomaly_threshold = mean + (2 * std)
        anomalies = engagement_trend[engagement_trend['Engagements'] > anomaly_threshold]
        if not anomalies.empty:
            anomaly = anomalies.iloc[0]
            st.markdown('<div class="anomaly-card">', unsafe_allow_html=True)
            st.markdown("<h3>⚠️ Peringatan Anomali Terdeteksi!</h3>", unsafe_allow_html=True)
            st.markdown(f"Lonjakan keterlibatan terdeteksi pada **{anomaly['Date']}**.")
            if st.button("Jelaskan Anomali Ini", key="anomaly_btn"):
                with st.spinner("Menganalisis..."):
                    prompt = f"Terdeteksi anomali keterlibatan pada {anomaly['Date']} ({anomaly['Engagements']} engagements). Berikan 3 kemungkinan penyebab dan 2 rekomendasi."
                    st.session_state.anomaly_insight = get_ai_insight(prompt)
            if st.session_state.anomaly_insight:
                st.markdown(f'<div class="insight-box">{st.session_state.anomaly_insight}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Tampilan Grafik
    chart_cols = st.columns(2)
    charts_to_display = [
        {"key": "sentiment", "title": "Analisis Sentimen", "col": chart_cols[0]},
        {"key": "trend", "title": "Tren Keterlibatan", "col": chart_cols[1]},
        {"key": "platform", "title": "Keterlibatan per Platform", "col": chart_cols[0]},
        {"key": "mediaType", "title": "Distribusi Jenis Media", "col": chart_cols[1]}, 
    ]
    
    for chart in charts_to_display:
        with chart["col"]:
            st.markdown(f'<div class="chart-container"><h3>{chart["title"]}</h3>', unsafe_allow_html=True)
            fig = None
            if not filtered_df.empty:
                if chart["key"] == "sentiment":
                    sentiment_data = filtered_df['Sentiment'].value_counts().reset_index()
                    fig = px.pie(sentiment_data, names='Sentiment', values='count', color_discrete_map={'Positive': '#08c792', 'Neutral': '#a1a1aa', 'Negative': '#ff5252'}, hole=.5)
                elif chart["key"] == "trend":
                    trend_data = filtered_df.groupby(filtered_df['Date'].dt.date)['Engagements'].sum().reset_index()
                    fig = px.line(trend_data, x='Date', y='Engagements')
                    fig.update_traces(line=dict(color='#4facfe', width=3))
                elif chart["key"] == "platform":
                    platform_data = filtered_df.groupby('Platform')['Engagements'].sum().sort_values(ascending=False).reset_index()
                    fig = px.bar(platform_data, x='Engagements', y='Platform', orientation='h', color_discrete_sequence=['#00f2fe'])
                elif chart["key"] == "mediaType":
                    media_type_data = filtered_df['Media Type'].value_counts().reset_index()
                    fig = px.bar(media_type_data, x='Media Type', y='count', color_discrete_sequence=['#4facfe'])
            
            if fig:
                st.session_state.chart_figures[chart["key"]] = fig 
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='var(--secondary-text-color)', legend_title_text='', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Tidak ada data untuk filter ini.")
            
            st.markdown('</div>', unsafe_allow_html=True)

    # Unduh Laporan
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("<h3>📄 Unduh Laporan Analisis</h3>", unsafe_allow_html=True)
    st.write("Unduh laporan lengkap dalam format HTML setelah analisis selesai.")
    
    if st.button("Buat & Unduh Laporan", key="download_html_btn", type="primary", use_container_width=True):
        st.error("Fitur unduh laporan saat ini sedang dalam pengembangan.")
    st.markdown('</div>', unsafe_allow_html=True)
