"""
Created for "MINI-PROJECT: | Data - Create a Machine Learning dashboard using Streamlit (Week 29)
Run with: streamlit run App.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Page configuration (no emoji)
st.set_page_config(
    page_title="Sakila Movie Rental Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# THEME MANAGEMENT
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    if 'theme_applied' not in st.session_state:
        st.session_state.theme_applied = False
    if 'custom_css' not in st.session_state:
        st.session_state.custom_css = ""

def get_theme_css(theme_colors):
    """Generate custom CSS based on theme colors"""
    return f"""
    <style>
    /* Main container styling */
    .stApp {{
        background: linear-gradient(135deg, {theme_colors['bg_start']} 0%, {theme_colors['bg_end']} 100%);
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {theme_colors['sidebar_start']} 0%, {theme_colors['sidebar_end']} 100%);
        border-right: 2px solid {theme_colors['accent']};
    }}
    
    /* Headers */
    h1, h2, h3 {{
        color: {theme_colors['title']} !important;
        font-weight: bold !important;
    }}
    
    /* Main header special */
    .main-header {{
        font-size: 2.5rem;
        color: {theme_colors['title']};
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }}
    
    .sub-header {{
        font-size: 1.5rem;
        color: {theme_colors['subtitle']};
        margin-top: 1rem;
        border-left: 4px solid {theme_colors['accent']};
        padding-left: 1rem;
    }}
    
    /* Info boxes */
    .info-box {{
        background: linear-gradient(135deg, {theme_colors['box_start']} 0%, {theme_colors['box_end']} 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid {theme_colors['accent']};
        color: {theme_colors['text']};
    }}
    
    .success-box {{
        background: linear-gradient(135deg, {theme_colors['success_start']} 0%, {theme_colors['success_end']} 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid {theme_colors['accent_success']};
        color: {theme_colors['text']};
    }}
    
    /* Formula styling */
    .formula {{
        font-family: monospace;
        background-color: {theme_colors['code_bg']};
        color: {theme_colors['code_text']};
        padding: 0.5rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.1rem;
    }}
    
    /* Metric cards */
    [data-testid="stMetricValue"] {{
        color: {theme_colors['metric_value']} !important;
        font-size: 1.8rem !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {theme_colors['metric_label']} !important;
    }}
    
    /* Button styling */
    .stButton > button {{
        background: linear-gradient(90deg, {theme_colors['button_start']} 0%, {theme_colors['button_end']} 100%);
        color: white !important;
        border: none;
        transition: transform 0.2s;
    }}
    
    .stButton > button:hover {{
        transform: scale(1.02);
        opacity: 0.95;
    }}
    
    /* Slider styling */
    .stSlider > div {{
        color: {theme_colors['accent']};
    }}
    
    /* Selectbox styling */
    .stSelectbox > div {{
        background-color: {theme_colors['select_bg']};
    }}
    
    /* Text color */
    .stMarkdown, p, li, label {{
        color: {theme_colors['text']};
    }}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
        color: {theme_colors['text']};
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
    }}
    
    /* Expander styling */
    .streamlit-expanderHeader {{
        color: {theme_colors['title']};
        background-color: {theme_colors['expander_bg']};
    }}
    
    /* Dataframe styling */
    .dataframe {{
        background-color: {theme_colors['table_bg']};
        color: {theme_colors['text']};
    }}
    
    /* Tree chart styling */
    .tree-container {{
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #ff6b6b;
        overflow-x: auto;
    }}
    .tree-title {{
        text-align: center;
        color: #ff6b6b;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 15px;
        font-family: monospace;
    }}
    .tree-node {{
        text-align: center;
        padding: 10px;
        margin: 5px;
    }}
    .tree-level {{
        display: flex;
        justify-content: center;
        margin: 5px 0;
    }}
    .tree-box {{
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border-radius: 8px;
        padding: 12px 20px;
        margin: 5px;
        border-left: 3px solid #ff6b6b;
        color: #f8f9fa;
        font-family: monospace;
        font-size: 0.85rem;
        min-width: 200px;
    }}
    .tree-arrow {{
        text-align: center;
        color: #ff6b6b;
        font-size: 1.2rem;
        margin: 5px 0;
    }}
    .formula-box {{
        background-color: #1e1e1e;
        border-radius: 6px;
        padding: 12px;
        margin: 10px 0;
        font-family: monospace;
        text-align: center;
        border: 1px solid #ff6b6b;
    }}
    .step-number {{
        color: #ff6b6b;
        font-weight: bold;
        font-size: 1rem;
    }}
    </style>
    """

def show_theme_settings():
    """Display theme configuration panel in sidebar (collapsible)"""
    with st.sidebar.expander("THEME SETTINGS", expanded=False):
        st.markdown("### Color Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            bg_start = st.color_picker("Background Start", "#667eea")
            bg_end = st.color_picker("Background End", "#764ba2")
            sidebar_start = st.color_picker("Sidebar Start", "#2c3e50")
            sidebar_end = st.color_picker("Sidebar End", "#3498db")
            title_color = st.color_picker("Title Color", "#ffffff")
        
        with col2:
            subtitle_color = st.color_picker("Subtitle Color", "#f0f0f0")
            accent_color = st.color_picker("Accent Color", "#ff6b6b")
            text_color = st.color_picker("Text Color", "#f8f9fa")
            metric_value = st.color_picker("Metric Value", "#00ff88")
            metric_label = st.color_picker("Metric Label", "#cccccc")
        
        st.markdown("### Box Colors")
        col3, col4 = st.columns(2)
        
        with col3:
            box_start = st.color_picker("Info Box Start", "#2d3748")
            box_end = st.color_picker("Info Box End", "#1a202c")
            success_start = st.color_picker("Success Box Start", "#22543d")
            success_end = st.color_picker("Success Box End", "#064e3b")
        
        with col4:
            code_bg = st.color_picker("Code Background", "#1e1e1e")
            code_text = st.color_picker("Code Text", "#d4d4d4")
            button_start = st.color_picker("Button Start", "#f093fb")
            button_end = st.color_picker("Button End", "#f5576c")
        
        theme_colors = {
            'bg_start': bg_start, 'bg_end': bg_end,
            'sidebar_start': sidebar_start, 'sidebar_end': sidebar_end,
            'title': title_color, 'subtitle': subtitle_color,
            'accent': accent_color, 'text': text_color,
            'metric_value': metric_value, 'metric_label': metric_label,
            'box_start': box_start, 'box_end': box_end,
            'success_start': success_start, 'success_end': success_end,
            'accent_success': '#22c55e',
            'code_bg': code_bg, 'code_text': code_text,
            'button_start': button_start, 'button_end': button_end,
            'select_bg': '#1e1e1e', 'expander_bg': '#2d3748',
            'table_bg': '#1e1e1e'
        }
        
        if st.button("Apply Theme", use_container_width=True):
            st.session_state.custom_css = get_theme_css(theme_colors)
            st.session_state.theme_applied = True
            st.rerun()
        
        if st.button("Reset to Default", use_container_width=True):
            default_colors = {
                'bg_start': '#667eea', 'bg_end': '#764ba2',
                'sidebar_start': '#2c3e50', 'sidebar_end': '#3498db',
                'title': '#ffffff', 'subtitle': '#f0f0f0',
                'accent': '#ff6b6b', 'text': '#f8f9fa',
                'metric_value': '#00ff88', 'metric_label': '#cccccc',
                'box_start': '#2d3748', 'box_end': '#1a202c',
                'success_start': '#22543d', 'success_end': '#064e3b',
                'code_bg': '#1e1e1e', 'code_text': '#d4d4d4',
                'button_start': '#f093fb', 'button_end': '#f5576c'
            }
            st.session_state.custom_css = get_theme_css(default_colors)
            st.session_state.theme_applied = True
            st.rerun()

# Initialize session state for theme
init_session_state()

# Show theme settings (always in sidebar, collapsible)
show_theme_settings()

# Apply custom CSS if theme was applied
if st.session_state.custom_css:
    st.markdown(st.session_state.custom_css, unsafe_allow_html=True)

# ============================================================================
# DATA LOADING AND CACHING
# ============================================================================

# Define file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# If CSV files are in same directory as app.py, use this instead:
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_daily_rentals():
    file_path = os.path.join(DATA_DIR, 'daily_rentals.csv')
    return pd.read_csv(file_path, parse_dates=['rental_day'])

@st.cache_data
def load_benefit():
    file_path = os.path.join(DATA_DIR, 'benefit.csv')
    return pd.read_csv(file_path)

@st.cache_data
def load_top_movies():
    file_path = os.path.join(DATA_DIR, 'all_top_movies.csv')
    return pd.read_csv(file_path)

@st.cache_data
def load_films():
    file_path = os.path.join(DATA_DIR, 'films.csv')
    return pd.read_csv(file_path)

@st.cache_resource
def load_model():
    """Loads the all-MiniLM-L6-v2 model from the sentence-transformers library. Unlike keyword matching or bag‑of‑words, this model captures the meaning of the description."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def create_embeddings(films_df, _model):
    """Create embeddings for all film descriptions. Takes the list of all movie descriptions, converts each into a 384‑dimension vector, and returns a NumPy array of shape (n_movies, 384)."""
    descriptions = films_df['description'].fillna('').tolist()
    embeddings = _model.encode(descriptions, show_progress_bar=False)       # Convert all movie descriptions to vectors
    return embeddings

# Load all data
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Home", "Exploratory Data Analysis", "Movie Prediction"])

try:
    daily_rentals = load_daily_rentals()
    benefit = load_benefit()
    all_top_movies = load_top_movies()
    films_df = load_films()
    model = load_model()
    embeddings = create_embeddings(films_df, model)

    st.sidebar.success(f"Data loaded successfully | {len(films_df)} movies available")

except FileNotFoundError as e:
    st.error(f"""
    Data files not found. Please run the Jupyter notebook first to create the CSV files.

    Error details: {e}

    Required files:
    - daily_rentals.csv
    - benefit.csv
    - all_top_movies.csv
    - films.csv
    """)
    st.stop()
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

# ==================== HOME PAGE ====================
if page == "Home":
    st.title("Sakila Movie Rental Dashboard")
    st.markdown("---")

    # Add an image as required
    st.image(
        "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=1200",
        caption="Movie Rental Analytics Dashboard",
        use_container_width=True
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## Dashboard Overview

        This professional dashboard provides comprehensive analytics for the Sakila DVD rental database,
        a sample database from MySQL simulating a movie rental store operation.

        ### Key Features

        **Exploratory Data Analysis**
        - Daily rental trends analysis for 2005
        - Store performance comparison
        - Top rented movies identification

        **Movie Prediction Engine**
        - AI-powered movie similarity search
        - Natural language description matching
        - Real-time similarity scoring

        ### Business Value
        - Identify peak rental periods
        - Optimize inventory allocation
        - Enhance customer recommendation systems
        """)

    with col2:
        st.markdown("### Key Metrics")

        total_rentals = daily_rentals['rental_count'].sum()
        total_revenue = benefit['total_benefit'].sum()
        total_movies = films_df.shape[0]
        avg_daily_rentals = daily_rentals['rental_count'].mean()

        metrics_style = """
        <div style='background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
            <p style='color: #666; font-size: 14px; margin: 0;'>{label}</p>
            <p style='color: #2c3e50; font-size: 28px; font-weight: bold; margin: 5px 0 0 0;'>{value}</p>
        </div>
        """

        st.markdown(metrics_style.format(label="Total Rentals (2005)", value=f"{total_rentals:,}"), unsafe_allow_html=True)
        st.markdown(metrics_style.format(label="Total Revenue", value=f"${total_revenue:,.2f}"), unsafe_allow_html=True)
        st.markdown(metrics_style.format(label="Movies Available", value=f"{total_movies:,}"), unsafe_allow_html=True)
        st.markdown(metrics_style.format(label="Average Daily Rentals", value=f"{avg_daily_rentals:.1f}"), unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("Data Source Information"):
        st.markdown("""
        **Database**: Sakila Sample Database
        **Source**: MySQL
        **Time Period**: 2005 rental data
        **Tables Used**: rental, payment, inventory, film, store, film_text

        The Sakila database models a DVD rental store, including information about films, customers,
        rentals, payments, and store operations.
        """)

# ==================== EDA PAGE ====================
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.markdown("---")

    # Section 1: Daily Rental Trends
    st.subheader("Daily Rental Trends by Store (2005)")

    fig1, ax1 = plt.subplots(figsize=(12, 6))

    colors = ['#2E86AB', '#A23B72']
    for idx, store in enumerate(sorted(daily_rentals['store_id'].unique())):
        store_data = daily_rentals[daily_rentals['store_id'] == store]
        ax1.plot(store_data['rental_day'], store_data['rental_count'],
                label=f'Store {store}', linewidth=2, color=colors[idx])

    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Number of Rentals', fontsize=12)
    ax1.set_title('Daily Rental Activity - 2005', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig1)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Store 1 Statistics**")
        store1_data = daily_rentals[daily_rentals['store_id'] == 1]['rental_count']
        st.markdown(f"""
        - Mean daily rentals: {store1_data.mean():.1f}
        - Peak daily rentals: {store1_data.max()}
        - Total rentals: {store1_data.sum():,}
        """)

    with col2:
        st.markdown("**Store 2 Statistics**")
        store2_data = daily_rentals[daily_rentals['store_id'] == 2]['rental_count']
        st.markdown(f"""
        - Mean daily rentals: {store2_data.mean():.1f}
        - Peak daily rentals: {store2_data.max()}
        - Total rentals: {store2_data.sum():,}
        """)

    st.markdown("---")

    # Section 2: Store Benefit Analysis
    st.subheader("Store Performance Analysis")

    col1, col2 = st.columns([3, 2])

    with col1:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        bars = ax2.bar(benefit['store_id'].astype(str), benefit['total_benefit'],
                       color=['#2E86AB', '#A23B72'], edgecolor='black', linewidth=1.5)

        ax2.set_xlabel('Store', fontsize=12)
        ax2.set_ylabel('Total Revenue (USD)', fontsize=12)
        ax2.set_title('Total Revenue Generated by Each Store', fontsize=14, fontweight='bold')

        for bar, value in zip(bars, benefit['total_benefit']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'${value:,.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

        plt.tight_layout()
        st.pyplot(fig2)

    with col2:
        st.markdown("### Revenue Breakdown")

        store1_rev = benefit[benefit['store_id'] == 1]['total_benefit'].values[0]
        store2_rev = benefit[benefit['store_id'] == 2]['total_benefit'].values[0]
        total_rev = store1_rev + store2_rev

        st.markdown(f"""
        **Store 1**
        - Revenue: ${store1_rev:,.2f}
        - Percentage: {(store1_rev/total_rev)*100:.1f}%

        **Store 2**
        - Revenue: ${store2_rev:,.2f}
        - Percentage: {(store2_rev/total_rev)*100:.1f}%

        **Total Revenue**: ${total_rev:,.2f}
        """)

        revenue_diff = abs(store1_rev - store2_rev)
        better_store = "Store 1" if store1_rev > store2_rev else "Store 2"
        st.info(f"Performance Insight: {better_store} generated ${revenue_diff:,.2f} more in revenue.")

    st.markdown("---")

    # Section 3: Top Movies Analysis
    st.subheader("Top 5 Most Rented Movies by Store (2005)")

    top5_store1 = all_top_movies[all_top_movies['store_id'] == 1].head(5)
    top5_store2 = all_top_movies[all_top_movies['store_id'] == 2].head(5)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Store 1")
        st.dataframe(
            top5_store1[['title', 'rating', 'rental_count']].reset_index(drop=True),
            use_container_width=True,
            height=300
        )

    with col2:
        st.markdown("#### Store 2")
        st.dataframe(
            top5_store2[['title', 'rating', 'rental_count']].reset_index(drop=True),
            use_container_width=True,
            height=300
        )

    with st.expander("Advanced Analytics"):
        st.markdown("### Movie Rating Distribution")
        rating_counts = all_top_movies.groupby('rating')['rental_count'].sum().sort_values(ascending=False)

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        rating_counts.plot(kind='bar', ax=ax3, color='#2E86AB', edgecolor='black')
        ax3.set_xlabel('Movie Rating', fontsize=12)
        ax3.set_ylabel('Total Rentals', fontsize=12)
        ax3.set_title('Most Popular Movie Ratings', fontsize=14, fontweight='bold')
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig3)

        most_popular_rating = rating_counts.index[0]
        most_popular_count = rating_counts.iloc[0]
        st.markdown(f"""
        **Key Insight**: Movies rated '{most_popular_rating}' are the most popular among customers,
        accounting for {most_popular_count:,} rentals across both stores.
        """)

    st.markdown("---")
    st.caption(f"Data Analysis Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==================== PREDICTION PAGE ====================
elif page == "Movie Prediction":
    st.title("Movie Similarity Prediction Engine")
    st.markdown("---")

    st.markdown("""
    This AI-powered tool analyzes movie descriptions and finds the most similar movies in our database.
    Enter a description below to receive personalized movie recommendations.
    """)

    with st.expander("How to Use This Tool"):
        st.markdown("""
        **Step 1**: Enter a movie description in the text area below

        **Step 2**: Click the 'Get Your Prediction' button

        **Step 3**: Review the top 3 most similar movies with their match scores

        **Tips for Best Results**:
        - Describe the plot, genre, or themes you're interested in
        - Include key elements like characters, settings, or moods
        - Be specific for better matching results
        """)

    user_input = st.text_area(
        "Movie Description Input",
        placeholder="Example: A young boy discovers he has magical powers and is invited to attend a school of witchcraft and wizardry...",
        height=150
    )

    if st.button("Get Your Prediction", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Processing description and finding matches..."):
                try:
                    user_embedding = model.encode([user_input])         # The user’s typed description is encoded into the same 384‑dimension vector space as the movies.
                    similarities = cosine_similarity(user_embedding, embeddings)[0]     # Calculates the cosine similarity between the user’s embedding vector and every movie embedding vector
                    top_indices = similarities.argsort()[-3:][::-1]     # the three most relevant movies to present to the user

                    st.success("Analysis complete. Found matching movies.")
                    st.markdown("---")
                    st.subheader("Recommended Movies")
                    #----------------------------
                    # Extracts the movie title, rating, and a short description preview. Converts the similarity score to a percentage and displays it prominently.
                    # --------------------------

                    for i, idx in enumerate(top_indices, 1):
                        movie = films_df.iloc[idx]
                        similarity_score = similarities[idx] * 100

                        with st.container():
                            col1, col2 = st.columns([4, 1])

                            with col1:
                                st.markdown(f"**{i}. {movie['Title']}**")
                                st.markdown(f"*Rating: {movie['Rating']}*")
                                desc_preview = movie['description'][:200] if len(movie['description']) > 200 else movie['description']
                                st.markdown(desc_preview)

                            with col2:
                                st.markdown("**Match Score**")
                                st.markdown(f"<h2 style='color: #2E86AB;'>{similarity_score:.1f}%</h2>", unsafe_allow_html=True)

                            st.markdown("---")

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
        else:
            st.warning("Please enter a movie description before searching.")

    with st.expander("Sample Movie Descriptions"):
        st.markdown("""
        **Action Adventure**
        "A fearless archaeologist travels the world searching for ancient artifacts while battling rival treasure hunters."

        **Drama**
        "A struggling musician faces personal challenges while pursuing his dream of performing at a major concert venue."

        **Science Fiction**
        "A group of astronauts travel through a wormhole to find a new habitable planet for humanity."

        **Comedy**
        "A group of mismatched friends embark on a chaotic road trip that tests their friendship."

        **Documentary**
        "An in-depth exploration of climate change impacts on coastal communities around the world."
        """)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Database Size**")
        st.markdown(f"<p style='font-size: 24px; font-weight: bold;'>{len(films_df)}</p>", unsafe_allow_html=True)
        st.markdown("Movies Available")

    with col2:
        st.markdown("**Rating Categories**")
        st.markdown(f"<p style='font-size: 24px; font-weight: bold;'>{films_df['rating'].nunique()}</p>", unsafe_allow_html=True)
        st.markdown("Unique Ratings")

    with col3:
        st.markdown("**Technology**")
        st.markdown("<p style='font-size: 24px; font-weight: bold;'>AI-Powered</p>", unsafe_allow_html=True)
        st.markdown("Sentence Transformers")