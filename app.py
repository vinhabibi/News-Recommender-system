import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from news_recommender_colab import (
    DataEngineeringPipeline,
    ContentAndContextualEngine,
    UserBasedCollaborativeFilter,
    HybridNewsRecommender,
    EvaluationAndEthicsSuite,
)

st.set_page_config(page_title="Hybrid News Recommender", layout="wide")
st.title("Personalized Hybrid News Recommendation Engine")
st.caption("Content-Based + User-Based Collaborative + Contextual Re-Ranking (EB-NeRD Architecture)")

@st.cache_resource
def load_system():
    pipe = DataEngineeringPipeline(data_dir="./data")
    pipe.extract_and_load()
    pipe.build_nlp_representations()
    train_df, val_df, test_df, cold_df, cold_users = pipe.create_splits()
    cc = ContentAndContextualEngine(pipe)
    cc.build_user_profiles(train_df, cold_users)
    cf = UserBasedCollaborativeFilter(pipe)
    cf.fit(train_df, cold_users)
    hybrid = HybridNewsRecommender(pipe, cc, cf)
    return pipe, cc, hybrid, sorted(list(cold_users))

pipe, cc, hybrid, cold_users = load_system()

with st.sidebar:
    st.header("Context & User Controls")
    warm_users = sorted(list(cc.user_profiles.keys()))[:50]
    user_mode = st.radio("User Profile Type", ["Warm User (With History)", "Cold-Start User (New)"])
    selected_user = st.selectbox("Select User ID", warm_users if user_mode.startswith("Warm") else cold_users)
    sim_hour = st.slider("Time of Day (Hour)", 0, 23, 8)
    location = st.selectbox("Reader Location", ["National", "Copenhagen", "Aarhus", "Odense", "Aalborg"])
    diversity_on = st.checkbox("Enable Filter-Bubble Diversity Re-Ranking", value=True)
    top_k = st.slider("Top-K Articles", 5, 15, 8)

base_dt = pipe.articles_df["published_time"].max().replace(hour=sim_hour, minute=0)
recs = hybrid.recommend(
    user_id=selected_user,
    current_time=base_dt,
    user_location=location,
    top_k=top_k,
    apply_diversity_rerank=diversity_on,
)

audit = EvaluationAndEthicsSuite.run_ethics_and_privacy_audit(recs, selected_user)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Recommendation Mode", recs.iloc[0]["mode"] if len(recs) else "N/A")
c2.metric("Anonymized User Hash", audit["anonymized_user_hash"])
c3.metric("Categories Surfaced", audit["unique_categories_surfaced"])
c4.metric("Privacy & Diversity", audit["filter_bubble_status"])

st.subheader(f"Top-{top_k} Recommended Articles ({sim_hour:02d}:00 · {location})")
st.dataframe(recs, use_container_width=True)
