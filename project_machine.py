import html
import streamlit as st
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD


st.set_page_config(page_title="AI Movie Recommender", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button {
        width: 100%; border-radius: 5px; height: 3em;
        background-color: #B20710; color: white; font-weight: bold; border: none;
    }
    .stButton>button:hover { background-color: #E50914; color: white; }
    h1 { color: #B20710; text-align: center; font-family: 'Arial Black', sans-serif; }

    .movie-card {
        border-radius: 10px;
        background-color: #1a1c23;
        padding: 10px;
        text-align: center;
        height: 100%;
        transition: transform 0.3s ease, border 0.3s ease;
        border: 1px solid transparent;
    }

    .movie-card:hover {
        transform: scale(1.08);
        border: 1px solid #B20710;
        z-index: 10;
        box-shadow: 0px 10px 20px rgba(178, 7, 16, 0.4);
    }

    label { color: white !important; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def build_ml_pipeline():

    df = pd.read_csv('movies_cleaned.csv')

    tfidf        = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['tags'])         

    cos_sim = cosine_similarity(tfidf_matrix).astype(np.float32)   

    knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=7)
    knn_model.fit(tfidf_matrix)

    N_COMPONENTS = 100                                      
    svd          = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    latent       = svd.fit_transform(tfidf_matrix)         
    svd_sim      = cosine_similarity(latent).astype(np.float32)    

    return df, cos_sim, knn_model, svd_sim, tfidf_matrix


df, cos_sim, knn_model, svd_sim, tfidf_matrix = build_ml_pipeline()


def fetch_poster(movie_id: int) -> str:
    url = (
        f"https://api.themoviedb.org/3/movie/{movie_id}"
        f"?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    )
    try:
        data = requests.get(url, timeout=5).json()
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except Exception:
        return "https://via.placeholder.com/500x750?text=No+Poster"


def recommend_cosine(movie_idx: int, top_k: int = 6):
    scores  = list(enumerate(cos_sim[movie_idx]))
    scores  = sorted(scores, key=lambda x: x[1], reverse=True)
    top     = scores[1: top_k + 1]                         
    indices = [i       for i, _ in top]
    sims    = [round(float(s), 4) for _, s in top]
    return indices, sims


def recommend_knn(movie_idx: int, top_k: int = 6):
    query           = tfidf_matrix[movie_idx]
    distances, nbrs = knn_model.kneighbors(query, n_neighbors=top_k + 1)
    indices = list(nbrs[0][1:])                            
    sims    = [round(1 - float(d), 4) for d in distances[0][1:]]
    return indices, sims


def recommend_svd(movie_idx: int, top_k: int = 6):
    scores  = list(enumerate(svd_sim[movie_idx]))
    scores  = sorted(scores, key=lambda x: x[1], reverse=True)
    top     = scores[1: top_k + 1]
    indices = [i       for i, _ in top]
    sims    = [round(float(s), 4) for _, s in top]
    return indices, sims


MODEL_FN = {
    "Cosine Similarity":          recommend_cosine,
    "KNN":                        recommend_knn,
    "SVD (Matrix Factorization)": recommend_svd,
}


def compute_metrics(sims: list) -> dict:
    return {
        "Average Similarity": round(float(np.mean(sims)), 4),
        "Maximum Similarity": round(float(np.max(sims)), 4),
    }



def plot_heatmap(rec_indices: list, rec_titles: list) -> plt.Figure:
    sub   = cos_sim[np.ix_(rec_indices, rec_indices)]
    lbls  = [t[:14] + "…" if len(t) > 14 else t for t in rec_titles]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        sub, annot=True, fmt=".2f",
        xticklabels=lbls, yticklabels=lbls,
        cmap='Reds', ax=ax,
        linewidths=0.5, linecolor='#0e1117'
    )
    ax.set_xlabel("Recommended Movies")
    ax.set_ylabel("Recommended Movies")
    plt.xticks(rotation=45, ha='right', color='white')
    plt.yticks(color='white')
    plt.tight_layout()
    return fig


def plot_bar_chart(rec_sims: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        [f"#{i + 1}" for i in range(len(rec_sims))],
        rec_sims,
        color='#B20710', edgecolor='#E50914', linewidth=0.8
    )
    for bar, val in zip(bars, rec_sims):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.005, f"{val:.3f}",
            ha='center', va='bottom', fontsize=8, color='white'
        )
    ax.set_xlabel("Recommendation Rank")
    ax.set_ylabel("Similarity Score")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig


def show_sidebar_analysis(rec_indices: list, rec_sims: list,
                           rec_titles: list, model_name: str):
    st.sidebar.subheader(" Model Performance")
    m = compute_metrics(rec_sims)
    st.sidebar.success(f"Model: {model_name}")
    st.sidebar.info(f"Average Similarity: {m['Average Similarity']}")
    st.sidebar.warning(f"Maximum Similarity: {m['Maximum Similarity']}")

    st.sidebar.write("###  Similarity Flow")
    for rank, (title, score) in enumerate(zip(rec_titles, rec_sims), start=1):
        st.sidebar.markdown(
            f"**#{rank}** &nbsp; `{score:.4f}` &nbsp; — &nbsp; {title}"
        )

    st.sidebar.write("###  Similarity Heatmap")
    st.sidebar.pyplot(plot_heatmap(rec_indices, rec_titles))

    st.sidebar.write("###  Similarity Bar Chart")
    st.sidebar.pyplot(plot_bar_chart(rec_sims))

st.markdown("<h1>AI Movie Recommender</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Visualization")
    show_analysis = st.checkbox("Show AI Analysis (Sidebar)")

    model_choice = st.radio(
        "Model:",
        ["Cosine Similarity", "KNN", "SVD (Matrix Factorization)"]
    )

selected_movie = st.selectbox("Select a movie you like:", df['title'].values)

if st.button('Show Recommendations'):
    idx = df[df['title'] == selected_movie].index[0]

    rec_indices, rec_sims = MODEL_FN[model_choice](idx, top_k=6)
    rec_titles            = df.iloc[rec_indices]['title'].tolist()

    st.markdown(
        f"### Results for: {html.escape(selected_movie)}",
        unsafe_allow_html=True,
    )
    cols = st.columns(6)
    for i, (movie_df_idx, sim_score) in enumerate(zip(rec_indices, rec_sims)):
        with cols[i]:
            movie_id    = df.iloc[movie_df_idx].movie_id
            safe_title  = html.escape(df.iloc[movie_df_idx].title)
            poster_url  = fetch_poster(movie_id)
            st.markdown(
                f'''
                <div class="movie-card">
                    <img src="{poster_url}"
                         style="width:100%; border-radius:5px;"
                         onerror="this.src='https://via.placeholder.com/500x750?text=No+Poster'">
                    <p style="color:white; font-size:0.8rem; margin-top:10px; font-weight:bold;">
                        {safe_title}
                    </p>
                    <p style="color:#B20710; font-size:0.75rem; margin:0;">
                        Score: {sim_score:.3f}
                    </p>
                </div>
                ''',
                unsafe_allow_html=True,
            )

    if show_analysis:
        show_sidebar_analysis(rec_indices, rec_sims, rec_titles, model_choice)
