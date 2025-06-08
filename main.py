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
    initial_sidebar_state="collapsed" # Mengubah sidebar menjadi collapsed by default
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

def get_ai_insight(prompt, model_name='gemini-1.5-flash'):
    """
    Memanggil API GenAI untuk menghasilkan wawasan berdasarkan prompt dan model yang diberikan.
    """
    if not configure_gemini_api():
        return "Gagal membuat wawasan: API tidak terkonfigurasi."

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            st.error(f"Model {model_name} tidak menghasilkan teks yang valid.")
            return "Gagal membuat wawasan. Silakan coba lagi."
    except Exception as e:
        st.error(f"Error saat memanggil model {model_name}: {e}.")
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
            insight_text_v1 = insights_for_chart.get("gemini-1.5-flash", "Belum ada wawasan yang dibuat.")
            insight_text_v2 = insights_for_chart.get("llama-3.3-8b-instruct", "Belum ada wawasan yang dibuat.")
            insight_text_v3 = insights_for_chart.get("gemini-1.5-pro-latest", "Belum ada wawasan yang dibuat.")


            if fig:
                fig_for_export = go.Figure(fig)
                fig_for_export.update_layout(
                    paper_bgcolor='#FFFFFF',
                    plot_bgcolor='#FFFFFF',
                    font_color='#333333'
                )
                
                try:
                    img_bytes = pio.to_image(fig_for_export, format="png", width=900, height=550, scale=1.5)
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    
                    chart_figures_html_sections += f"""
                    <div class="insight-sub-section">
                        <h3>{chart_title}</h3>
                        <img src="data:image/png;base64,{img_base64}" alt="{chart_title}" style="max-width: 100%; height: auto; display: block; margin: 10px auto; border: 1px solid #ddd; border-radius: 5px;">
                        <h4>Wawasan AI (gemini-1.5-flash):</h4>
                        <div class="insight-box">{insight_text_v1}</div>
                        <h4>Wawasan AI (llama-3.3-8b-instruct):</h4>
                        <div class="insight-box">{insight_text_v2}</div>
                        <h4>Wawasan AI (gemini-1.5-pro-latest):</h4>
                        <div class="insight-box">{insight_text_v3}</div>
                    </div>
                    """
                except Exception as e:
                    chart_figures_html_sections += f"""
                    <div class="insight-sub-section">
                        <h3>{chart_title}</h3>
                        <p>Gagal menyertakan grafik ini (Error: {e}).</p>
                    </div>
                    """
            else:
                 if any(insight.strip() != "Belum ada wawasan yang dibuat." for insight in [insight_text_v1, insight_text_v2, insight_text_v3]):
                    chart_figures_html_sections += f"""
                    <div class="insight-sub-section">
                        <h3>{chart_title}</h3>
                        <p>Tidak ada grafik yang tersedia untuk {chart_title}.</p>
                        <h4>Wawasan AI (gemini-1.5-flash):</h4>
                        <div class="insight-box">{insight_text_v1}</div>
                        <h4>Wawasan AI (llama-3.3-8b-instruct):</h4>
                        <div class="insight-box">{insight_text_v2}</div>
                        <h4>Wawasan AI (gemini-1.5-pro-latest):</h4>
                        <div class="insight-box">{insight_text_v3}</div>
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
            .insight-box {{ background-color: #e9ecef; padding: 10px; border-radius: 5px; font-size: 0.9em; white-space: pre-wrap; word-wrap: break-word; }}
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
    return html_content.encode('utf-8')

def load_css():
    """Menyuntikkan CSS kustom untuk gaya visual tingkat lanjut."""
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400..900&display=swap');
            body { background-color: #0f172a !important; }
            .stApp {
                background-image: radial-gradient(at top left, #1e293b, #0f172a, black);
                color: #cbd5e1;
            }
            .main-header { font-family: 'Orbitron', sans-serif; text-align: center; margin-bottom: 2rem; }
            .main-header h1 {
                background: -webkit-linear-gradient(45deg, #06B6D4, #6366F1);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.75rem; font-weight: 900;
            }
            .main-header p { color: #94a3b8; font-size: 1.1rem; }
            [data-testid="stSidebar"] {
                background-color: rgba(15, 23, 42, 0.6);
                backdrop-filter: blur(10px);
                border-right: 1px solid #334155;
            }
            .chart-container, .insight-hub, .anomaly-card {
                border: 1px solid #475569;
                background-color: rgba(30, 41, 59, 0.6);
                backdrop-filter: blur(15px);
                border-radius: 1rem; padding: 1.5rem;
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
                margin-bottom: 2rem; box-sizing: border-box;
            }
            .anomaly-card { border: 2px solid #f59e0b; background-color: rgba(245, 158, 11, 0.1); }
            .insight-box {
                background-color: rgba(15, 23, 42, 0.7);
                border: 1px solid #334155;
                border-radius: 0.5rem; padding: 1rem; margin-top: 1rem;
                min-height: 150px; white-space: pre-wrap;
                word-wrap: break-word; overflow-wrap: break-word; font-size: 0.9rem;
            }
            .chart-container h3, .insight-hub h3, .anomaly-card h3, .insight-hub h4 {
                color: #5eead4; margin-top: 0; margin-bottom: 1rem;
                display: flex; align-items: center; gap: 0.5rem; font-weight: 600;
            }
            .uploaded-file-info {
                background-color: rgba(30, 41, 59, 0.6);
                border: 1px solid #475569;
                border-radius: 1rem; padding: 1.5rem; margin-bottom: 2rem;
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1); color: #cbd5e1;
            }
            .uploaded-file-info h3 { color: #5eead4; margin-top: 0; margin-bottom: 1rem; }
            .uploaded-file-info p { margin-bottom: 0.5rem; }
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
if 'show_analysis' not in st.session_state: st.session_state.show_analysis = False

# Tampilan unggah file
if st.session_state.data is None: 
    with st.container():
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("### ☁️ Unggah File CSV Anda")
            st.write("Pastikan file memiliki kolom 'Date', 'Engagements', 'Sentiment', 'Platform', 'Media_Type', 'Location', dan (opsional) 'Headline'.")
            uploaded_file = st.file_uploader("Pilih file CSV", type="csv", key="main_file_uploader")
            if uploaded_file is not None:
                if uploaded_file.name != st.session_state.last_uploaded_file_name or uploaded_file.size != st.session_state.last_uploaded_file_size:
                    st.session_state.data = parse_csv(uploaded_file)
                    if st.session_state.data is not None:
                        st.session_state.last_uploaded_file_name = uploaded_file.name
                        st.session_state.last_uploaded_file_size = uploaded_file.size
                        st.session_state.show_analysis = False
                        st.rerun() 
            st.markdown('</div>', unsafe_allow_html=True)

# Tampilan Dasbor Utama
if st.session_state.data is not None:
    df = st.session_state.data

    st.markdown(f"""
        <div class="uploaded-file-info">
            <h3>☁️ File Terunggah</h3>
            <p><strong>Nama File:</strong> {st.session_state.last_uploaded_file_name}</p>
            <p><strong>Ukuran File:</strong> {st.session_state.last_uploaded_file_size / (1024 * 1024):.2f} MB</p>
            <p style="color: #5eead4; font-weight: bold;">File CSV berhasil diunggah dan diproses!</p>
    """, unsafe_allow_html=True)
    
    if st.button("Hapus File Terunggah", key="clear_file_btn"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if not st.session_state.show_analysis:
        if st.button("Lihat Hasil Analisis Datamu!", key="show_analysis_btn", use_container_width=True, type="primary"):
            st.session_state.show_analysis = True
            st.rerun()

    if st.session_state.show_analysis:
        
        with st.expander("⚙️ Filter Data & Opsi Tampilan", expanded=True):
            # --- UX UPDATE: Logic untuk 'All' di dalam multiselect ---
            def multiselect_with_all(label, options, key):
                all_option = "Pilih Semua"
                options_with_all = [all_option] + options
                
                # Cek state sebelum widget dirender
                prev_selection = st.session_state.get(key, [])

                # Render widget
                selection = st.multiselect(label, options=options_with_all, key=f"ms_{key}")

                # Cek state setelah widget dirender untuk deteksi perubahan
                if prev_selection != selection:
                    if all_option in selection and all_option not in prev_selection:
                        # "All" baru saja dipilih
                        st.session_state[key] = options_with_all
                    elif all_option not in selection and all_option in prev_selection:
                        # "All" baru saja tidak dipilih
                        st.session_state[key] = []
                    else:
                        # Pilihan individual diubah
                        st.session_state[key] = [s for s in selection if s != all_option]
                    st.rerun()
                
                final_selection = [s for s in st.session_state.get(key, []) if s != all_option]
                return final_selection

            min_date, max_date = df['Date'].min().date(), df['Date'].max().date()

            filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 3])
            
            with filter_col1:
                platform_options = sorted(list(df['Platform'].unique()))
                platform = multiselect_with_all("Platform", platform_options, "platform_filter")

                media_type_options = sorted(list(df['Media Type'].unique()))
                media_type = multiselect_with_all("Media Type", media_type_options, "media_type_filter")
            
            with filter_col2:
                sentiment_options = sorted(list(df['Sentiment'].unique()))
                sentiment = multiselect_with_all("Sentiment", sentiment_options, "sentiment_filter")
                
                location_options = sorted(list(df['Location'].unique()))
                location = multiselect_with_all("Location", location_options, "location_filter")

            with filter_col3:
                date_range = st.date_input("Pilih Rentang Tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date, format="DD/MM/YYYY")
                start_date, end_date = date_range if len(date_range) == 2 else (min_date, max_date)

            filter_state = f"{''.join(sorted(platform))}{''.join(sorted(sentiment))}{''.join(sorted(media_type))}{''.join(sorted(location))}{start_date}{end_date}"
            if 'last_filter_state' not in st.session_state or st.session_state.last_filter_state != filter_state:
                st.session_state.chart_insights, st.session_state.campaign_summary, st.session_state.post_idea, st.session_state.anomaly_insight, st.session_state.chart_figures = {}, "", "", "", {}
                st.session_state.last_filter_state = filter_state

        filtered_df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
        if platform: filtered_df = filtered_df[filtered_df['Platform'].isin(platform)]
        if sentiment: filtered_df = filtered_df[filtered_df['Sentiment'].isin(sentiment)]
        if media_type: filtered_df = filtered_df[filtered_df['Media Type'].isin(media_type)]
        if location: filtered_df = filtered_df[filtered_df['Location'].isin(location)]

        # --- Tampilan Utama ---
        
        # Deteksi Anomali (jika ada)
        engagement_trend = filtered_df.groupby(filtered_df['Date'].dt.date)['Engagements'].sum().reset_index()
        if len(engagement_trend) > 7:
            mean, std = engagement_trend['Engagements'].mean(), engagement_trend['Engagements'].std()
            anomaly_threshold = mean + (2 * std) if std > 0 else mean + 0.1
            anomalies = engagement_trend[engagement_trend['Engagements'] > anomaly_threshold]
            if not anomalies.empty:
                anomaly = anomalies.iloc[0]
                st.markdown('<div class="anomaly-card">', unsafe_allow_html=True)
                st.markdown("<h3>⚠️ Peringatan Anomali Terdeteksi!</h3>", unsafe_allow_html=True)
                st.markdown(f"Kami mendeteksi lonjakan keterlibatan yang tidak biasa pada **{anomaly['Date']}** dengan **{int(anomaly['Engagements']):,}** keterlibatan (rata-rata: {int(mean):,}).")
                if st.button("✨ Jelaskan Anomali Ini", key="anomaly_btn"):
                    with st.spinner("Menganalisis penyebab anomali..."):
                        anomaly_day_data = filtered_df[filtered_df['Date'].dt.date == anomaly['Date']]
                        top_headlines = ', '.join(anomaly_day_data.nlargest(3, 'Engagements')['Headline'].tolist())
                        prompt = f"Anda adalah analis data. Terdeteksi anomali keterlibatan pada {anomaly['Date']} sebesar {anomaly['Engagements']}. Rata-rata adalah {mean:.2f}. Judul berita teratas: {top_headlines}. Berikan 3 kemungkinan penyebab dan 2 rekomendasi tindakan."
                        st.session_state.anomaly_insight = get_ai_insight(prompt)
                if st.session_state.anomaly_insight:
                    st.markdown(f'<div class="insight-box">{st.session_state.anomaly_insight}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Tampilan Grafik
        chart_cols = st.columns(2)
        charts_to_display = [{"key": "sentiment", "title": "Analisis Sentimen", "col": chart_cols[0]}, {"key": "trend", "title": "Tren Keterlibatan", "col": chart_cols[1]}, {"key": "platform", "title": "Keterlibatan per Platform", "col": chart_cols[0]}, {"key": "mediaType", "title": "Distribusi Jenis Media", "col": chart_cols[1]}, {"key": "location", "title": "5 Lokasi Teratas", "col": chart_cols[0]}]
        
        def get_chart_prompt(key, data_json, model_name):
            base_prompts = {
                "sentiment": f"Data distribusi sentimen: {data_json}.", "trend": f"Data tren keterlibatan harian: {data_json}.",
                "platform": f"Data keterlibatan per platform: {data_json}.", "mediaType": f"Data distribusi jenis media: {data_json}.",
                "location": f"Data keterlibatan per lokasi: {data_json}."
            }
            personas = {
                "gemini-1.5-flash": "Anda adalah konsultan media. Berikan 3 wawasan taktis dan dapat ditindaklanjuti dari data ini. Format sebagai daftar bernomor.",
                "llama-3.3-8b-instruct": "Anda adalah seorang brand strategist. Fokus pada peluang dan risiko yang tersembunyi. Berikan 3 wawasan alternatif yang berbeda dari analisis standar. Format sebagai daftar bernomor.",
                "gemini-1.5-pro-latest": "Anda adalah analis pasar futuristik. Analisis data ini untuk implikasi jangka panjang (6-12 bulan). Identifikasi potensi pergeseran pasar atau tren disrupsi. Berikan 2-3 wawasan visioner. Format sebagai daftar bernomor."
            }
            return f"{personas.get(model_name, '')} {base_prompts.get(key, '')}"

        for chart in charts_to_display:
            with chart["col"]:
                st.markdown(f'<div class="chart-container" key="chart-{chart["key"]}">', unsafe_allow_html=True)
                st.markdown(f'<h3>{chart["title"]}</h3>', unsafe_allow_html=True)
                
                fig, chart_data_for_prompt = (None, None)
                try:
                    if chart["key"] == "sentiment":
                        sentiment_data = filtered_df['Sentiment'].value_counts().reset_index()
                        sentiment_data.columns = ['Sentiment', 'count']
                        if not sentiment_data.empty: fig = px.pie(sentiment_data, names='Sentiment', values='count', color_discrete_sequence=px.colors.qualitative.Pastel)
                        chart_data_for_prompt = sentiment_data.to_json()
                    elif chart["key"] == "trend":
                        trend_data = filtered_df.groupby(filtered_df['Date'].dt.date)['Engagements'].sum().reset_index()
                        if not trend_data.empty: fig = px.line(trend_data, x='Date', y='Engagements', labels={'Date': 'Tanggal', 'Engagements': 'Total Keterlibatan'})
                        chart_data_for_prompt = trend_data.tail(10).to_json()
                    elif chart["key"] == "platform":
                        platform_data = filtered_df.groupby('Platform')['Engagements'].sum().sort_values(ascending=False).reset_index()
                        if not platform_data.empty: fig = px.bar(platform_data, x='Platform', y='Engagements', color='Platform')
                        chart_data_for_prompt = platform_data.to_json()
                    elif chart["key"] == "mediaType":
                        media_data = filtered_df['Media Type'].value_counts().reset_index()
                        if not media_data.empty: fig = px.pie(media_data, names='Media Type', values='count', hole=.3)
                        chart_data_for_prompt = media_data.to_json()
                    elif chart["key"] == "location": 
                        location_data = filtered_df.groupby('Location')['Engagements'].sum().nlargest(5).reset_index()
                        if not location_data.empty: fig = px.bar(location_data, y='Location', x='Engagements', orientation='h')
                        chart_data_for_prompt = location_data.to_json()
                except Exception: pass # Fail silently if chart can't be created

                if fig: 
                    st.session_state.chart_figures[chart["key"]] = fig 
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#cbd5e1', legend_title_text='')
                    st.plotly_chart(fig, use_container_width=True)

                    selected_insight_version = st.selectbox("Pilih Model AI", options=["gemini-1.5-flash", "llama-3.3-8b-instruct", "gemini-1.5-pro-latest"], key=f"sel_{chart['key']}")
                    
                    if st.button("✨ Generate AI Insight", key=f"btn_{chart['key']}"):
                        with st.spinner(f"Menganalisis {chart['title']} dengan 3 model AI..."):
                            if chart_data_for_prompt:
                                st.session_state.chart_insights[chart['key']] = {
                                    model: get_ai_insight(get_chart_prompt(chart['key'], chart_data_for_prompt, model), model)
                                    for model in ["gemini-1.5-flash", "llama-3.3-8b-instruct", "gemini-1.5-pro-latest"]
                                }
                            st.rerun() 

                    current_insights = st.session_state.chart_insights.get(chart['key'], {})
                    insight_text_to_display = current_insights.get(selected_insight_version, "Klik 'Generate AI Insight' untuk menghasilkan wawasan.")
                    st.markdown(f'<div class="insight-box">{insight_text_to_display}</div>', unsafe_allow_html=True)
                else:
                    st.warning("Tidak ada data yang tersedia untuk grafik ini dengan filter yang dipilih.")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Pusat Wawasan AI
        st.markdown('<div class="insight-hub">', unsafe_allow_html=True)
        st.markdown("<h3>🧠 Pusat Wawasan AI</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4>⚡ Ringkasan Strategi Kampanye</h4>", unsafe_allow_html=True)
            if st.button("Buat Ringkasan", key="summary_btn", use_container_width=True):
                with st.spinner("Membuat ringkasan strategi..."):
                    prompt = f"Anda adalah konsultan strategi media. Analisis data ini: Total sebutan: {len(filtered_df)}, Rata-rata keterlibatan: {filtered_df['Engagements'].mean():.2f}, Distribusi Sentimen: {filtered_df['Sentiment'].value_counts().to_json()}, Keterlibatan per Platform: {filtered_df.groupby('Platform')['Engagements'].sum().to_json()}. Berikan ringkasan eksekutif dan 3 rekomendasi strategis."
                    st.session_state.campaign_summary = get_ai_insight(prompt)
            st.markdown(f'<div class="insight-box">{st.session_state.campaign_summary or "Klik untuk membuat ringkasan strategis."}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("<h4>💡 Generator Ide Konten AI</h4>", unsafe_allow_html=True)
            if st.button("✨ Buat Ide Postingan", key="idea_btn", use_container_width=True):
                with st.spinner("Mencari ide terbaik..."):
                    if not filtered_df.empty:
                        best_platform = filtered_df.groupby('Platform')['Engagements'].sum().idxmax()
                        top_headlines = ', '.join(filtered_df[filtered_df['Platform'] == best_platform].nlargest(5, 'Engagements')['Headline'].tolist())
                    else: best_platform, top_headlines = "N/A", "N/A"
                    prompt = f"Anda adalah ahli strategi media sosial. Platform terbaik: {best_platform}. Topik populer: {top_headlines}. Buat satu contoh postingan (termasuk saran visual & tagar) untuk platform tersebut dalam Bahasa Indonesia."
                    st.session_state.post_idea = get_ai_insight(prompt)
            st.markdown(f'<div class="insight-box">{st.session_state.post_idea or "Klik untuk menghasilkan ide postingan."}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Unduh Laporan
        st.markdown("---")
        st.markdown("<h3>📄 Unduh Laporan Analisis</h3>", unsafe_allow_html=True)
        if st.button("Unduh Laporan HTML", key="download_html_btn", type="primary", use_container_width=True):
            with st.spinner("Membangun laporan HTML..."):
                html_data = generate_html_report(st.session_state.campaign_summary, st.session_state.post_idea, st.session_state.anomaly_insight, st.session_state.chart_insights, st.session_state.chart_figures, charts_to_display)
                if html_data:
                    st.download_button(label="Klik untuk Mengunduh", data=html_data, file_name="Laporan_Media_Intelligence.html", mime="text/html", key="actual_download_button_html")
                    st.success("Laporan siap! Gunakan fitur cetak browser untuk menyimpan sebagai PDF.")
