import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import io
import base64
import plotly.io as pio

# --- PAGE CONFIGURATION & STYLING ---
# Set page configuration. This must be the first Streamlit command.
st.set_page_config(
    page_title="Media Intelligence Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CORE FUNCTIONS & LOGIC ---

def configure_gemini_api():
    """
    Configures the Gemini API using an API key.
    This key is hardcoded for demonstration purposes.
    In production, it's best to use st.secrets or environment variables.
    """
    # It's recommended to use st.secrets for managing API keys securely
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except (FileNotFoundError, KeyError):
        st.warning("Gemini API key not found in st.secrets. Please add it to your secrets.toml file.")
        return False

    if not api_key:
        st.warning("Gemini API Key is missing. Some AI features may not function.")
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}. Please ensure the API Key is valid.")
        return False

def get_ai_insight(prompt):
    """
    Calls the Gemini API to generate insights based on a given prompt.
    Uses the 'gemini-1.5-flash' model.
    """
    if not genai.API_KEY:
        st.error("AI insight generation failed: Gemini API is not configured.")
        return "Failed to generate insight: API not configured."

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            st.error("Gemini API did not return valid text. The response was unexpected.")
            return "Failed to generate insight. Please try again."
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}. Please check your API key and internet connection.")
        return "Failed to generate insight: A connection or API error occurred."

def generate_html_report(campaign_summary, post_idea, anomaly_insight, chart_insights, chart_figures_dict, charts_to_display_info):
    """
    Creates an HTML report from the AI-generated insights and charts.
    `chart_figures_dict` is a dictionary {chart_key: plotly_figure_object}.
    `charts_to_display_info` is a list of chart info to get full titles.
    """
    current_date = pd.Timestamp.now().strftime("%d %B %Y, %H:%M")

    anomaly_section_html = ""
    if anomaly_insight and anomaly_insight.strip() != "No insight generated yet.":
        anomaly_section_html = f"""
        <div class="section">
            <h2>3. Anomaly Insight</h2>
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
            insight_text_v1 = insights_for_chart.get("gemini-1.5-flash", "No insight generated yet.")
            insight_text_v2 = insights_for_chart.get("alternative-perspective", "No insight generated yet.")

            if fig:
                fig_for_export = go.Figure(fig)
                fig_for_export.update_layout(
                    paper_bgcolor='#FFFFFF',
                    plot_bgcolor='#FFFFFF',
                    font_color='#1d1d1f',
                    legend_font_color='#1d1d1f',
                    title_font_color='#1d1d1f'
                )

                try:
                    img_bytes = pio.to_image(fig_for_export, format="png", width=900, height=500, scale=2)
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

                    chart_figures_html_sections += f"""
                    <div class="chart-section">
                        <h3>{chart_title}</h3>
                        <img src="data:image/png;base64,{img_base64}" alt="{chart_title}" class="chart-image">
                        <h4>AI Insight (Primary)</h4>
                        <div class="insight-box">{insight_text_v1}</div>
                        <h4>AI Insight (Alternative)</h4>
                        <div class="insight-box">{insight_text_v2}</div>
                    </div>
                    """
                except Exception as e:
                    chart_figures_html_sections += f"<p>Failed to include chart: {chart_title} (Error: {e}).</p>"
            else:
                 chart_figures_html_sections += f"""
                    <div class="chart-section">
                        <h3>{chart_title}</h3>
                         <p>No chart available for {chart_title}.</p>
                        <h4>AI Insight (Primary)</h4>
                        <div class="insight-box">{insight_text_v1}</div>
                        <h4>AI Insight (Alternative)</h4>
                        <div class="insight-box">{insight_text_v2}</div>
                    </div>
                    """
    else:
        chart_figures_html_sections = "<p>No insights or charts have been generated yet.</p>"


    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Media Intelligence Report</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                line-height: 1.6;
                color: #1d1d1f;
                background-color: #f5f5f7;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                max-width: 1000px;
                margin: 40px auto;
                background-color: #ffffff;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            }}
            h1, h2, h3, h4 {{
                color: #000000;
                font-weight: 600;
                margin-top: 1.5em;
                margin-bottom: 0.8em;
            }}
            h1 {{ font-size: 2.5rem; font-weight: 700; margin-bottom: 0.2em; }}
            .subtitle {{ color: #6e6e73; margin-top: 0; margin-bottom: 2em; }}
            .section, .chart-section {{
                border-top: 1px solid #e5e5e5;
                padding-top: 20px;
                margin-top: 20px;
            }}
            .insight-box {{
                background-color: #f5f5f7;
                padding: 15px;
                border-radius: 12px;
                font-size: 0.95em;
                white-space: pre-wrap;
                word-wrap: break-word;
                color: #333333;
            }}
            .chart-image {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 20px auto;
                border: 1px solid #e5e5e5;
                border-radius: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Media Intelligence Report</h1>
            <p class="subtitle">Generated on: {current_date}</p>

            <div class="section">
                <h2>1. Campaign Strategy Summary</h2>
                <div class="insight-box">{campaign_summary if campaign_summary else "No summary has been generated."}</div>
            </div>

            <div class="section">
                <h2>2. AI Content Idea</h2>
                <div class="insight-box">{post_idea if post_idea else "No post idea has been generated."}</div>
            </div>

            {anomaly_section_html}

            <div class="section">
                <h2>4. Chart Insights</h2>
                {chart_figures_html_sections}
            </div>
        </div>
    </body>
    </html>
    """
    return html_content.encode('utf-8')

def load_apple_style_css():
    """Injects custom CSS for an Apple-inspired UI."""
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

            /* --- Base & Body --- */
            html, body, .stApp {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                background-color: #f5f5f7 !important;
                color: #1d1d1f;
            }

            /* --- Main Container --- */
            .main-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 2rem 1rem;
            }

            /* --- Header --- */
            .main-header {
                text-align: left;
                margin-bottom: 3rem;
            }
            .main-header h1 {
                font-size: 2.5rem; /* 40px */
                font-weight: 700;
                color: #000000;
                margin-bottom: 0.25rem;
            }
            .main-header p {
                font-size: 1.125rem; /* 18px */
                color: #6e6e73;
                margin-top: 0;
            }

            /* --- Sidebar --- */
            [data-testid="stSidebar"] {
                background-color: #ffffff;
                border-right: 1px solid #d2d2d7;
                padding: 1.5rem;
            }
            [data-testid="stSidebar"] h3 {
                color: #000000;
                font-weight: 600;
                font-size: 1.2rem;
            }
            [data-testid="stSidebar"] .stSelectbox,
            [data-testid="stSidebar"] .stDateInput {
                margin-bottom: 1rem;
            }

            /* --- General Card Style --- */
            .card {
                background-color: #ffffff;
                border: 1px solid #d2d2d7;
                border-radius: 20px;
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.04);
            }
            .card h3, .card h4 {
                color: #000000;
                margin-top: 0;
                margin-bottom: 1rem;
                font-weight: 600;
            }
            .card h3 { font-size: 1.25rem; }
            .card h4 { font-size: 1rem; }

            /* --- Anomaly Card --- */
            .anomaly-card {
                border-color: #ff9f0a;
                background-color: #fff8eb;
            }
            .anomaly-card h3 { color: #bf7300; }

            /* --- Insight Box within Cards --- */
            .insight-box {
                background-color: #f5f5f7;
                border-radius: 12px;
                padding: 1rem;
                margin-top: 1rem;
                min-height: 120px;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-size: 0.95rem;
                color: #333333;
            }

            /* --- Buttons --- */
            .stButton > button {
                background-color: #007aff;
                color: white;
                border: none;
                border-radius: 12px;
                padding: 0.75rem 1.5rem;
                font-weight: 500;
                font-size: 1rem;
                transition: background-color 0.2s ease;
            }
            .stButton > button:hover {
                background-color: #005ecb;
                color: white;
            }
            .stButton > button:focus {
                box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.3);
                outline: none;
            }
            /* Secondary Button (e.g., Clear File) */
            .stButton[key="clear_file_btn"] > button {
                background-color: #e5e5ea;
                color: #007aff;
            }
            .stButton[key="clear_file_btn"] > button:hover {
                background-color: #dcdce0;
            }

            /* --- Uploaded File Info --- */
            .uploaded-file-info {
                background-color: #e8f3ff;
                border: 1px solid #b3d7ff;
                border-radius: 20px;
                padding: 2rem;
                margin-bottom: 2rem;
            }
            .uploaded-file-info h3 { margin-top: 0; color: #005ecb; }

            /* --- Radio Buttons for Insight Version --- */
            div[data-testid="stRadio"] > label {
                padding: 0.5rem 1rem;
                margin: 0.25rem;
                border-radius: 10px;
                background-color: #f5f5f7;
                border: 1px solid #e5e5e5;
            }
            div[data-testid="stRadio"] > div[role="radiogroup"] {
                flex-direction: row !important;
                justify-content: flex-start;
                gap: 0.5rem;
            }

            /* --- Plotly Chart Styling --- */
            .stPlotlyChart {
                border-radius: 12px;
                overflow: hidden;
            }
        </style>
    """, unsafe_allow_html=True)

@st.cache_data
def parse_csv(uploaded_file):
    """Reads and cleans an uploaded CSV file into a pandas DataFrame."""
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
        st.error(f"Failed to process CSV file. Please ensure it's correctly formatted. Error: {e}")
        return None

# --- STREAMLIT UI ---
load_apple_style_css()
api_configured = configure_gemini_api()

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'chart_insights' not in st.session_state:
    st.session_state.chart_insights = {}
if 'campaign_summary' not in st.session_state:
    st.session_state.campaign_summary = ""
if 'post_idea' not in st.session_state:
    st.session_state.post_idea = ""
if 'anomaly_insight' not in st.session_state:
    st.session_state.anomaly_insight = ""
if 'chart_figures' not in st.session_state:
    st.session_state.chart_figures = {}
if 'last_uploaded_details' not in st.session_state:
    st.session_state.last_uploaded_details = None

# Main Header
st.markdown("""
    <div class="main-header">
        <h1>Media Intelligence</h1>
        <p>Your AI-powered analytics dashboard by Ryan Vandiaz Media Agency</p>
    </div>
""", unsafe_allow_html=True)


# File upload view (only shows if data hasn't been uploaded)
if st.session_state.data is None:
    with st.container():
        _, col2, _ = st.columns([1, 1.5, 1])
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Welcome. Start by uploading your data.")
            st.write("Please upload a CSV file with columns: 'Date', 'Engagements', 'Sentiment', 'Platform', 'Media_Type', 'Location', and 'Headline'.")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="main_file_uploader", label_visibility="collapsed")
            if uploaded_file is not None:
                current_file_details = (uploaded_file.name, uploaded_file.size)
                if current_file_details != st.session_state.last_uploaded_details:
                    st.session_state.data = parse_csv(uploaded_file)
                    if st.session_state.data is not None:
                        st.session_state.last_uploaded_details = current_file_details
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# Main Dashboard View (after file upload)
if st.session_state.data is not None:
    df = st.session_state.data

    # --- Sidebar Filters ---
    with st.sidebar:
        st.markdown("<h3>Data Filters</h3>", unsafe_allow_html=True)

        platform = st.selectbox("Platform", ["All"] + sorted(list(df['Platform'].unique())), key='platform_filter')
        sentiment = st.selectbox("Sentiment", ["All"] + sorted(list(df['Sentiment'].unique())), key='sentiment_filter')
        media_type = st.selectbox("Media Type", ["All"] + sorted(list(df['Media Type'].unique())), key='media_type_filter')
        location = st.selectbox("Location", ["All"] + sorted(list(df['Location'].unique())), key='location_filter')

        min_date, max_date = df['Date'].min().date(), df['Date'].max().date()
        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date, key='start_date_filter')
        end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date, key='end_date_filter')

        # Logic to reset insights if filters change
        filter_state = f"{platform}{sentiment}{media_type}{location}{start_date}{end_date}"
        if 'last_filter_state' not in st.session_state or st.session_state.last_filter_state != filter_state:
            st.session_state.chart_insights.clear()
            st.session_state.campaign_summary = ""
            st.session_state.post_idea = ""
            st.session_state.anomaly_insight = ""
            st.session_state.chart_figures.clear()
            st.session_state.last_filter_state = filter_state
            
        st.markdown("---")
        if st.button("Clear Uploaded File", key="clear_file_btn", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # Filter DataFrame based on sidebar controls
    filtered_df = df[
        (df['Date'].dt.date >= start_date) &
        (df['Date'].dt.date <= end_date) &
        (df['Platform'] == platform if platform != "All" else True) &
        (df['Sentiment'] == sentiment if sentiment != "All" else True) &
        (df['Media Type'] == media_type if media_type != "All" else True) &
        (df['Location'] == location if location != "All" else True)
    ]

    if filtered_df.empty:
        st.warning("No data matches the current filter settings. Please adjust the filters in the sidebar.")
    else:
        # --- AI Insight Hub ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🧠 AI Insight Hub", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h4>⚡ Campaign Strategy Summary</h4>", unsafe_allow_html=True)
            if st.button("Generate Summary", key="summary_btn", use_container_width=True):
                with st.spinner("Crafting strategic summary..."):
                    prompt = f"""
                    You are a senior media strategy consultant. Analyze the following campaign data comprehensively. Provide a 2-sentence executive summary followed by 3 impactful, actionable strategic recommendations.
                    Use the following data:
                    - Filtered data sample (first 5 rows): {filtered_df.head().to_json()}
                    - Total mentions: {len(filtered_df)}
                    - Average engagements: {filtered_df['Engagements'].mean():.2f}
                    - Sentiment Distribution: {filtered_df['Sentiment'].value_counts().to_json()}
                    - Engagements per Platform: {filtered_df.groupby('Platform')['Engagements'].sum().to_json()}
                    Focus on the big picture: What's the main story? Where are the biggest opportunities and key risks? Format your answer clearly and concisely.
                    """
                    st.session_state.campaign_summary = get_ai_insight(prompt)
            st.markdown(f'<div class="insight-box">{st.session_state.campaign_summary or "Click to generate a strategic summary of all data."}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("<h4>💡 AI Content Idea Generator</h4>", unsafe_allow_html=True)
            if st.button("✨ Create Post Idea", key="idea_btn", use_container_width=True):
                with st.spinner("Finding the best idea..."):
                    best_platform_series = filtered_df.groupby('Platform')['Engagements'].sum()
                    best_platform = best_platform_series.idxmax() if not best_platform_series.empty else "unknown"
                    top_posts = filtered_df.nlargest(5, 'Engagements')
                    top_headlines = ', '.join(top_posts['Headline'].tolist()) if not top_posts.empty else "no data"

                    prompt = f"""
                    You are a creative social media strategist. Based on the following data, create one sample post for the **{best_platform}** platform.
                    - Best Performing Platform: {best_platform}
                    - High-Performing Topics (from headlines): {top_headlines}
                    The post must:
                    1. Be written in Indonesian.
                    2. Have an engaging and platform-appropriate tone for {best_platform}.
                    3. Provide a clear visual concept suggestion.
                    4. Include 3-5 relevant hashtags.
                    Format the output clearly: "Platform:", "Post Content:", "Visual Suggestion:", and "Hashtags:".
                    """
                    st.session_state.post_idea = get_ai_insight(prompt)
            st.markdown(f'<div class="insight-box">{st.session_state.post_idea or "Click to generate a post idea based on your best-performing data."}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Anomaly Detection ---
        engagement_trend = filtered_df.groupby(filtered_df['Date'].dt.date)['Engagements'].sum().reset_index()
        if len(engagement_trend) > 7:
            mean = engagement_trend['Engagements'].mean()
            std = engagement_trend['Engagements'].std()
            anomaly_threshold = mean + (2 * std) if std > 0 else mean * 1.5
            anomalies = engagement_trend[engagement_trend['Engagements'] > anomaly_threshold]

            if not anomalies.empty:
                anomaly = anomalies.iloc[0]
                st.markdown('<div class="card anomaly-card">', unsafe_allow_html=True)
                st.markdown("<h3>⚠️ Anomaly Detected!</h3>", unsafe_allow_html=True)
                st.markdown(f"We detected an unusual engagement spike on **{pd.to_datetime(anomaly['Date']).strftime('%B %d, %Y')}** with **{int(anomaly['Engagements']):,}** engagements (period average: {int(mean):,}).")

                if st.button("✨ Explain This Anomaly", key="anomaly_btn", use_container_width=True):
                    with st.spinner("Analyzing cause of anomaly..."):
                        anomaly_date = pd.to_datetime(anomaly['Date']).date()
                        anomaly_day_data = filtered_df[filtered_df['Date'].dt.date == anomaly_date]
                        top_headlines_on_anomaly_day = ', '.join(anomaly_day_data.nlargest(3, 'Engagements')['Headline'].tolist())

                        prompt = f"""
                        You are a media intelligence data analyst. An engagement anomaly was detected on {anomaly_date}.
                        - Engagements on that day: {anomaly['Engagements']}
                        - Historical average engagement (from filtered data): {mean:.2f}
                        - Top contributing headlines on that day: {top_headlines_on_anomaly_day or "No specific headlines found."}
                        Provide 3 likely causes for this anomaly and 2 recommended actions. Format as a numbered list.
                        """
                        st.session_state.anomaly_insight = get_ai_insight(prompt)

                if st.session_state.anomaly_insight:
                    st.markdown(f'<div class="insight-box">{st.session_state.anomaly_insight}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # --- Chart Display ---
        chart_cols = st.columns(2)
        charts_to_display = [
            {"key": "trend", "title": "Engagement Trend", "col": chart_cols[0]},
            {"key": "platform", "title": "Engagement by Platform", "col": chart_cols[1]},
            {"key": "sentiment", "title": "Sentiment Analysis", "col": chart_cols[0]},
            {"key": "mediaType", "title": "Media Type Distribution", "col": chart_cols[1]},
            {"key": "location", "title": "Top 5 Locations", "col": chart_cols[0]},
        ]

        def get_chart_prompt(key, data_json, perspective="primary"):
            prompts = {
                "sentiment": "Based on this sentiment distribution data, provide 3 sharp, actionable insights for brand communication strategy. Format as a numbered list.",
                "trend": "From this daily engagement trend data, provide 3 strategic insights on peaks, dips, and general patterns. What does this mean for campaign rhythm? Format as a numbered list.",
                "platform": "Using this engagement per platform data, give 3 actionable insights. Identify 'champion' and 'opportunity' platforms. Format as a numbered list.",
                "mediaType": "From this media type distribution data, provide 3 strategic insights. Analyze audience preference based on content format. Format as a numbered list.",
                "location": "Based on this engagement per location data, give 3 geo-strategic insights. Identify key markets and emerging ones. Format as a numbered list."
            }
            base_prompt = f"You are a professional media intelligence consultant. {prompts.get(key, '')} Here is the data: {data_json}."
            if perspective == "alternative":
                return base_prompt + " Now, provide 3 alternative/complementary insights from a different perspective, or focus on hidden opportunities/overlooked risks. Avoid repeating ideas from the primary insight."
            return base_prompt

        for chart in charts_to_display:
            with chart["col"]:
                st.markdown(f'<div class="card" key="chart-{chart["key"]}">', unsafe_allow_html=True)
                st.markdown(f'<h3>{chart["title"]}</h3>', unsafe_allow_html=True)
                
                fig = None
                chart_data_for_prompt = None
                
                # Chart generation logic
                if chart["key"] == "sentiment":
                    data = filtered_df['Sentiment'].value_counts()
                    if not data.empty:
                        fig = px.pie(data, names=data.index, values=data.values, color_discrete_sequence=["#007aff", "#ff9f0a", "#30d158", "#ff3b30"])
                        chart_data_for_prompt = data.to_json()
                elif chart["key"] == "trend":
                    data = filtered_df.groupby(filtered_df['Date'].dt.date)['Engagements'].sum().reset_index()
                    if not data.empty:
                        fig = px.area(data, x='Date', y='Engagements', labels={'Date': 'Date', 'Engagements': 'Total Engagements'})
                        fig.update_traces(line=dict(color='#007aff', width=3), fillcolor='rgba(0,122,255,0.2)')
                        chart_data_for_prompt = data.tail(10).to_json(orient="records")
                elif chart["key"] == "platform":
                    data = filtered_df.groupby('Platform')['Engagements'].sum().sort_values(ascending=False)
                    if not data.empty:
                        fig = px.bar(data, x=data.index, y=data.values, labels={'x': 'Platform', 'y': 'Total Engagements'}, color=data.index)
                        chart_data_for_prompt = data.to_json()
                elif chart["key"] == "mediaType":
                    data = filtered_df['Media Type'].value_counts()
                    if not data.empty:
                        fig = px.pie(data, names=data.index, values=data.values, hole=.4)
                        chart_data_for_prompt = data.to_json()
                elif chart["key"] == "location":
                    data = filtered_df.groupby('Location')['Engagements'].sum().nlargest(5).sort_values(ascending=True)
                    if not data.empty:
                        fig = px.bar(data, y=data.index, x=data.values, orientation='h', labels={'y': 'Location', 'x': 'Total Engagements'}, color=data.index)
                        chart_data_for_prompt = data.to_json()

                if fig:
                    st.session_state.chart_figures[chart["key"]] = fig
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#1d1d1f',
                        legend_title_text='',
                        margin=dict(l=0, r=0, t=40, b=0),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                else:
                    st.write("No data available for this chart with the selected filters.")
                    st.session_state.chart_figures[chart["key"]] = None

                # AI Insight Generation for each chart
                if st.button(f"✨ Generate AI Insight", key=f"insight_btn_{chart['key']}", use_container_width=True):
                    with st.spinner(f"Analyzing {chart['title']}..."):
                        if chart_data_for_prompt:
                            prompt_v1 = get_chart_prompt(chart['key'], chart_data_for_prompt, "primary")
                            insight_v1 = get_ai_insight(prompt_v1)
                            prompt_v2 = get_chart_prompt(chart['key'], chart_data_for_prompt, "alternative")
                            insight_v2 = get_ai_insight(prompt_v2)
                            st.session_state.chart_insights[chart['key']] = {
                                "gemini-1.5-flash": insight_v1,
                                "alternative-perspective": insight_v2
                            }
                        else:
                            no_data_msg = "Not enough data to generate insights."
                            st.session_state.chart_insights[chart['key']] = {
                                "gemini-1.5-flash": no_data_msg, "alternative-perspective": no_data_msg
                            }
                
                # Display insight selector and box
                current_insights = st.session_state.chart_insights.get(chart['key'], {})
                if current_insights:
                    selected_insight_version = st.radio(
                        "Insight Perspective:",
                        ("gemini-1.5-flash", "alternative-perspective"),
                        format_func=lambda x: "Primary" if x == "gemini-1.5-flash" else "Alternative",
                        key=f"insight_selector_{chart['key']}",
                        label_visibility="collapsed"
                    )
                    insight_text_to_display = current_insights.get(selected_insight_version, "Click 'Generate AI Insight' to produce an analysis.")
                    st.markdown(f'<div class="insight-box">{insight_text_to_display}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # --- HTML Report Download Section ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📄 Download Full Report")
        st.write("Generate and download a comprehensive HTML report containing all generated insights and charts.")
        
        if st.button("Generate & Download Report", key="download_html_btn", type="primary", use_container_width=True):
            with st.spinner("Building HTML report with charts..."):
                chart_insights_for_report = {
                    chart_info["key"]: st.session_state.chart_insights.get(chart_info["key"], {}) 
                    for chart_info in charts_to_display
                }
                
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
                        label="Download Report Now",
                        data=html_data,
                        file_name="Media_Intelligence_Report.html",
                        mime="text/html",
                        key="actual_download_button_html",
                        use_container_width=True
                    )
                    st.success("Report is ready! You can also print the HTML file to PDF from your browser.")
                else:
                    st.error("Failed to generate HTML report. Ensure some data or insights exist.")
        st.markdown('</div>', unsafe_allow_html=True)
