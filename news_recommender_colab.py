# ==============================================================================
# HYBRID NEWS RECOMMENDATION SYSTEM (module: news_recommender_colab.py)
# Dataset: EB-NeRD Parquet Schema (articles.parquet, behaviors.parquet, history.parquet)
# ==============================================================================
# COLAB QUICKSTART:
# 1. Put news_recommender_colab.py and app.py in the same folder (e.g. /content).
# 2. !pip install -q pyarrow scikit-learn scipy streamlit pyngrok pandas numpy
# 3. (Optional) !unzip -q your_dataset.zip -d ./data
# 4. %run news_recommender_colab.py   # runs evaluation
# 5. !streamlit run app.py &>/content/logs.txt &  then  !npx localtunnel --port 8501
# ==============================================================================

import os
import re
import glob
import math
import hashlib
import zipfile
import warnings
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ==============================================================================
# TEAM 1: DATA ENGINEERING & PREPROCESSING (Members 1, 2, 3)
# ==============================================================================

class DataEngineeringPipeline:
    """
    Member 1: Data Acquisition & Schema Standardization (EB-NeRD Parquet Loader)
    Member 2: NLP & Text Preprocessing (Cleaning, Tokenization, TF-IDF + SVD)
    Member 3: Temporal Interaction Splitting & Cold-Start Holdout Setup
    """

    def __init__(self, data_dir: str = "./data", random_seed: int = 42):
        self.data_dir = data_dir
        self.rng = np.random.default_rng(random_seed)
        self.articles_df = None
        self.behaviors_df = None
        self.history_df = None
        self.tfidf_vectorizer = None
        self.svd_model = None
        self.article_tfidf = None
        self.article_dense = None
        self.article_id_to_idx = {}
        self.idx_to_article_id = {}

    def extract_and_load(self, zip_path: str = None):
        """Member 1: Unzips dataset if provided and loads parquet files."""
        if zip_path and os.path.exists(zip_path):
            print(f"[Member 1] Extracting {zip_path} -> {self.data_dir} ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.data_dir)

        art_files = glob.glob(f"{self.data_dir}/**/articles*.parquet", recursive=True)
        beh_files = glob.glob(f"{self.data_dir}/**/behavio*s*.parquet", recursive=True)
        his_files = glob.glob(f"{self.data_dir}/**/history*.parquet", recursive=True)

        if art_files and beh_files and his_files:
            print(f"[Member 1] Found Parquet Files:\n  Articles: {art_files[0]}\n  Behaviors: {beh_files[0]}\n  History: {his_files[0]}")
            self.articles_df = pd.read_parquet(art_files[0])
            self.behaviors_df = pd.read_parquet(beh_files[0])
            self.history_df = pd.read_parquet(his_files[0])
        else:
            print("[Member 1] Parquet files not found on disk. Generating EB-NeRD schema-compatible demo dataset...")
            self._generate_synthetic_ebnerd()

        self._standardize_schemas()
        return self.articles_df, self.behaviors_df, self.history_df

    def _generate_synthetic_ebnerd(self, n_articles: int = 600, n_users: int = 400, n_impressions: int = 2500):
        """Fallback generator matching exact EB-NeRD parquet schemas for instant testing."""
        categories = ["News", "Sports", "Finance", "Technology", "Entertainment", "Lifestyle", "Crime"]
        locations = ["Copenhagen", "Aarhus", "Odense", "Aalborg", "National"]
        base_time = datetime(2023, 5, 18, 8, 0, 0)

        topic_vocab = {
            "News": ["parliament election policy minister climate energy summit global treaty"],
            "Sports": ["championship football match goal striker tournament coach victory league"],
            "Finance": ["inflation interest rates stock market banking economy growth housing"],
            "Technology": ["artificial intelligence software cybersecurity cloud chip startup data"],
            "Entertainment": ["cinema festival concert album actor streaming series premiere"],
            "Lifestyle": ["nutrition fitness travel gastronomy wellness architecture design"],
            "Crime": ["investigation police court verdict security cyber fraud trial"],
        }

        articles = []
        for i in range(n_articles):
            cat = categories[i % len(categories)]
            pub_dt = base_time - timedelta(hours=int(self.rng.integers(1, 240)))
            words = topic_vocab[cat][0].split()
            sampled = " ".join(self.rng.choice(words, size=5, replace=True))
            articles.append({
                "article_id": 9000000 + i,
                "title": f"{cat} Update {i}: {sampled.title()}",
                "subtitle": f"In-depth coverage on {sampled} and regional impact.",
                "body": f"Detailed report analyzing {sampled}. Experts highlight key shifts in {cat.lower()}.",
                "category_str": cat,
                "published_time": pub_dt,
                "total_inviews": int(self.rng.integers(500, 50000)),
                "total_clicks": int(self.rng.integers(20, 4000)),
                "total_read_time": float(self.rng.uniform(1000, 90000)),
                "sentiment_label": self.rng.choice(["Positive", "Neutral", "Negative"]),
                "location": locations[i % len(locations)],
            })
        self.articles_df = pd.DataFrame(articles)

        art_ids = self.articles_df["article_id"].values
        art_cats = self.articles_df["category_str"].values

        histories = []
        user_fav_cat = {}
        for u in range(n_users):
            uid = 100000 + u
            fav = categories[u % len(categories)]
            user_fav_cat[uid] = fav
            fav_pool = art_ids[art_cats == fav]
            other_pool = art_ids[art_cats != fav]
            clicked_hist = list(self.rng.choice(fav_pool, size=min(8, len(fav_pool)), replace=False)) + \
                           list(self.rng.choice(other_pool, size=3, replace=False))
            hist_times = [base_time - timedelta(days=int(self.rng.integers(1, 20))) for _ in clicked_hist]
            histories.append({
                "user_id": uid,
                "article_id_fixed": clicked_hist,
                "impression_time_fixed": hist_times,
                "read_time_fixed": [float(self.rng.uniform(15, 180)) for _ in clicked_hist],
                "scroll_percentage_fixed": [float(self.rng.uniform(40, 100)) for _ in clicked_hist],
            })
        self.history_df = pd.DataFrame(histories)

        behaviors = []
        for imp_id in range(n_impressions):
            uid = 100000 + int(self.rng.integers(0, n_users))
            fav = user_fav_cat[uid]
            imp_dt = base_time + timedelta(hours=int(self.rng.integers(0, 168)))
            fav_pool = art_ids[art_cats == fav]
            other_pool = art_ids[art_cats != fav]
            pos_art = int(self.rng.choice(fav_pool))
            neg_arts = list(self.rng.choice(other_pool, size=7, replace=False))
            inview = [pos_art] + [int(x) for x in neg_arts]
            self.rng.shuffle(inview)
            behaviors.append({
                "impression_id": imp_id + 1,
                "user_id": uid,
                "impression_time": imp_dt,
                "article_ids_inview": inview,
                "article_ids_clicked": [pos_art],
                "device_type": int(self.rng.choice([1, 2, 3])),
                "is_subscriber": bool(self.rng.choice([True, False])),
                "location": locations[uid % len(locations)],
            })
        self.behaviors_df = pd.DataFrame(behaviors)

    def _standardize_schemas(self):
        """Member 1 & 3: Normalizes column names, timestamps, and missing values."""
        art = self.articles_df.copy()
        art["article_id"] = art["article_id"].astype(int)
        for col in ["title", "subtitle", "body", "category_str"]:
            if col not in art.columns:
                art[col] = ""
            art[col] = art[col].fillna("").astype(str)

        if "published_time" in art.columns:
            art["published_time"] = pd.to_datetime(art["published_time"], errors="coerce")
        else:
            art["published_time"] = pd.Timestamp("2023-05-15")
        art["published_time"] = art["published_time"].fillna(art["published_time"].median())

        if "total_inviews" not in art.columns:
            art["total_inviews"] = 1000
        if "total_clicks" not in art.columns:
            art["total_clicks"] = 50
        art["ctr"] = (art["total_clicks"].fillna(0) + 1.0) / (art["total_inviews"].fillna(0) + 20.0)
        if "location" not in art.columns:
            art["location"] = "National"

        self.articles_df = art.drop_duplicates(subset=["article_id"]).reset_index(drop=True)
        self.article_id_to_idx = {aid: idx for idx, aid in enumerate(self.articles_df["article_id"])}
        self.idx_to_article_id = {idx: aid for aid, idx in self.article_id_to_idx.items()}

        beh = self.behaviors_df.copy()
        beh["user_id"] = beh["user_id"].astype(int)
        beh["impression_time"] = pd.to_datetime(beh["impression_time"], errors="coerce")
        beh = beh.dropna(subset=["impression_time", "article_ids_inview", "article_ids_clicked"]).reset_index(drop=True)
        if "location" not in beh.columns:
            beh["location"] = "National"
        self.behaviors_df = beh

        his = self.history_df.copy()
        his["user_id"] = his["user_id"].astype(int)
        self.history_df = his.drop_duplicates(subset=["user_id"]).reset_index(drop=True)
        print(f"[Member 1] Standardized Schemas -> Articles: {len(self.articles_df):,}, "
              f"Behaviors: {len(self.behaviors_df):,}, Users w/ History: {len(self.history_df):,}")

    @staticmethod
    def _clean_text(text: str) -> str:
        """Member 2: Lowercasing, punctuation removal, and whitespace normalization."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def build_nlp_representations(self, max_features: int = 5000, svd_components: int = 64):
        """Member 2: Builds TF-IDF sparse matrix & TruncatedSVD dense semantic embeddings."""
        corpus = (
            self.articles_df["category_str"] + " " +
            self.articles_df["title"] + " " +
            self.articles_df["subtitle"]
        ).apply(self._clean_text)

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )
        self.article_tfidf = self.tfidf_vectorizer.fit_transform(corpus)

        n_comp = min(svd_components, self.article_tfidf.shape[1] - 1, self.article_tfidf.shape[0] - 1)
        self.svd_model = TruncatedSVD(n_components=max(8, n_comp), random_state=42)
        self.article_dense = self.svd_model.fit_transform(self.article_tfidf)
        norms = np.linalg.norm(self.article_dense, axis=1, keepdims=True) + 1e-9
        self.article_dense = self.article_dense / norms

        print(f"[Member 2] Built NLP Vectors -> TF-IDF shape: {self.article_tfidf.shape}, "
              f"Dense Embeddings shape: {self.article_dense.shape}")
        return self.article_tfidf, self.article_dense

    def create_splits(self, cold_user_ratio: float = 0.15):
        """Member 3: Temporal Train/Val/Test split + explicit Cold-Start User Holdout."""
        beh = self.behaviors_df.sort_values("impression_time").reset_index(drop=True)
        unique_users = beh["user_id"].unique()
        n_cold = max(1, int(len(unique_users) * cold_user_ratio))
        cold_users = set(self.rng.choice(unique_users, size=n_cold, replace=False))

        warm_beh = beh[~beh["user_id"].isin(cold_users)].reset_index(drop=True)
        cold_test_df = beh[beh["user_id"].isin(cold_users)].reset_index(drop=True)

        n_warm = len(warm_beh)
        train_end = int(n_warm * 0.70)
        val_end = int(n_warm * 0.85)

        train_df = warm_beh.iloc[:train_end].reset_index(drop=True)
        val_df = warm_beh.iloc[train_end:val_end].reset_index(drop=True)
        test_df = warm_beh.iloc[val_end:].reset_index(drop=True)

        print(f"[Member 3] Splits Created -> Train: {len(train_df):,}, Val: {len(val_df):,}, "
              f"Warm Test: {len(test_df):,}, Cold-Start Test: {len(cold_test_df):,} ({len(cold_users)} held-out users)")
        return train_df, val_df, test_df, cold_test_df, cold_users


# ==============================================================================
# TEAM 2: CONTENT-BASED & CONTEXTUAL MODELING (Members 4, 5, 6)
# ==============================================================================

class ContentAndContextualEngine:
    """
    Member 4: Content-Based Filtering Engine (User Profile Embedding + Cosine Sim)
    Member 5: Contextual Filtering Module (Time-of-Day, Recency Decay, Location)
    Member 6: Cold-Start Fallback & Real-Time Click Feedback Loop
    """

    def __init__(self, pipeline: DataEngineeringPipeline, decay_lambda: float = 0.015):
        self.pipe = pipeline
        self.decay_lambda = decay_lambda
        self.user_profiles = {}

    def build_user_profiles(self, train_df: pd.DataFrame, cold_users: set):
        """Member 4: Aggregates historical + training clicks into user interest vectors."""
        user_clicks = defaultdict(list)

        for _, row in self.pipe.history_df.iterrows():
            uid = int(row["user_id"])
            if uid in cold_users:
                continue
            arts = row.get("article_id_fixed", [])
            if arts is not None:
                user_clicks[uid].extend([int(a) for a in arts])

        for _, row in train_df.iterrows():
            uid = int(row["user_id"])
            if uid in cold_users:
                continue
            clicked = row.get("article_ids_clicked", [])
            if clicked is not None:
                user_clicks[uid].extend([int(a) for a in clicked])

        for uid, aids in user_clicks.items():
            valid_idxs = [self.pipe.article_id_to_idx[a] for a in aids[-25:] if a in self.pipe.article_id_to_idx]
            if not valid_idxs:
                continue
            weights = np.linspace(0.5, 1.0, len(valid_idxs))[:, None]
            vecs = self.pipe.article_dense[valid_idxs]
            profile = np.sum(vecs * weights, axis=0)
            profile = profile / (np.linalg.norm(profile) + 1e-9)
            self.user_profiles[uid] = profile

        print(f"[Member 4] Built Content User Profiles for {len(self.user_profiles):,} warm users.")

    def update_user_profile_online(self, user_id: int, clicked_article_id: int, alpha: float = 0.25):
        """Member 6: Real-time feedback loop updating user embedding upon a new click."""
        if clicked_article_id not in self.pipe.article_id_to_idx:
            return
        art_vec = self.pipe.article_dense[self.pipe.article_id_to_idx[clicked_article_id]]
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = art_vec.copy()
        else:
            updated = (1.0 - alpha) * self.user_profiles[user_id] + alpha * art_vec
            self.user_profiles[user_id] = updated / (np.linalg.norm(updated) + 1e-9)

    def score_content(self, user_id: int, candidate_article_ids: list) -> np.ndarray:
        """Member 4: Cosine similarity between user profile and candidate articles."""
        n = len(candidate_article_ids)
        if user_id not in self.user_profiles:
            return np.zeros(n, dtype=np.float32)

        u_vec = self.user_profiles[user_id]
        scores = np.zeros(n, dtype=np.float32)
        for i, aid in enumerate(candidate_article_ids):
            idx = self.pipe.article_id_to_idx.get(int(aid))
            if idx is not None:
                sim = float(np.dot(u_vec, self.pipe.article_dense[idx]))
                scores[i] = max(0.0, (sim + 1.0) / 2.0)
        return scores

    def score_context(self, candidate_article_ids: list, current_time: pd.Timestamp, user_location: str = "National") -> np.ndarray:
        """Member 5: Freshness decay + time-of-day category alignment + regional boost."""
        hour = current_time.hour
        if 5 <= hour < 11:
            preferred_cats = {"News", "Finance", "Technology"}
        elif 11 <= hour < 18:
            preferred_cats = {"Sports", "Technology", "News"}
        else:
            preferred_cats = {"Entertainment", "Lifestyle", "Sports"}

        scores = np.zeros(len(candidate_article_ids), dtype=np.float32)
        art_df = self.pipe.articles_df

        for i, aid in enumerate(candidate_article_ids):
            idx = self.pipe.article_id_to_idx.get(int(aid))
            if idx is None:
                continue
            row = art_df.iloc[idx]
            age_hours = max(0.0, (current_time - row["published_time"]).total_seconds() / 3600.0)
            freshness = math.exp(-self.decay_lambda * min(age_hours, 240.0))
            time_boost = 1.0 if row["category_str"] in preferred_cats else 0.4
            loc_boost = 1.0 if (user_location != "National" and row.get("location") == user_location) else 0.5
            scores[i] = 0.55 * freshness + 0.25 * time_boost + 0.20 * loc_boost
        return scores

    def score_cold_start_trending(self, candidate_article_ids: list, current_time: pd.Timestamp) -> np.ndarray:
        """Member 6: Trending velocity + CTR fallback for cold-start users & breaking news."""
        scores = np.zeros(len(candidate_article_ids), dtype=np.float32)
        art_df = self.pipe.articles_df
        for i, aid in enumerate(candidate_article_ids):
            idx = self.pipe.article_id_to_idx.get(int(aid))
            if idx is None:
                continue
            row = art_df.iloc[idx]
            age_hours = max(1.0, (current_time - row["published_time"]).total_seconds() / 3600.0)
            recency = math.exp(-0.02 * min(age_hours, 168.0))
            pop = min(1.0, math.log1p(row["total_clicks"]) / 10.0)
            scores[i] = 0.5 * pop + 0.3 * float(row["ctr"]) * 5.0 + 0.2 * recency
        return np.clip(scores, 0.0, 1.0)


# ==============================================================================
# TEAM 3: COLLABORATIVE FILTERING & HYBRID INTEGRATION (Members 7, 8, 9)
# ==============================================================================

class UserBasedCollaborativeFilter:
    """Member 7: Sparse User-Item Interaction Matrix & Cosine Neighborhood CF."""

    def __init__(self, pipeline: DataEngineeringPipeline, top_k_neighbors: int = 25):
        self.pipe = pipeline
        self.top_k = top_k_neighbors
        self.user_to_row = {}
        self.interaction_matrix = None
        self.user_sim_matrix = None

    def fit(self, train_df: pd.DataFrame, cold_users: set):
        rows, cols, vals = [], [], []
        active_users = sorted([u for u in self.pipe.history_df["user_id"].unique() if u not in cold_users])
        self.user_to_row = {u: idx for idx, u in enumerate(active_users)}

        for _, row in self.pipe.history_df.iterrows():
            uid = int(row["user_id"])
            if uid not in self.user_to_row:
                continue
            u_idx = self.user_to_row[uid]
            for aid in row.get("article_id_fixed", []):
                a_idx = self.pipe.article_id_to_idx.get(int(aid))
                if a_idx is not None:
                    rows.append(u_idx)
                    cols.append(a_idx)
                    vals.append(1.0)

        for _, row in train_df.iterrows():
            uid = int(row["user_id"])
            if uid not in self.user_to_row:
                continue
            u_idx = self.user_to_row[uid]
            for aid in row.get("article_ids_clicked", []):
                a_idx = self.pipe.article_id_to_idx.get(int(aid))
                if a_idx is not None:
                    rows.append(u_idx)
                    cols.append(a_idx)
                    vals.append(1.5)

        n_u = len(self.user_to_row)
        n_a = len(self.pipe.articles_df)
        self.interaction_matrix = csr_matrix((vals, (rows, cols)), shape=(n_u, n_a), dtype=np.float32)

        if n_u > 1:
            sim = cosine_similarity(self.interaction_matrix, dense_output=True)
            np.fill_diagonal(sim, 0.0)
            self.user_sim_matrix = sim
        else:
            self.user_sim_matrix = np.zeros((n_u, n_u), dtype=np.float32)

        print(f"[Member 7] Fitted User-Based CF Matrix: {self.interaction_matrix.shape} "
              f"(nnz={self.interaction_matrix.nnz:,})")

    def score_cf(self, user_id: int, candidate_article_ids: list) -> np.ndarray:
        n = len(candidate_article_ids)
        if user_id not in self.user_to_row or self.user_sim_matrix is None:
            return np.zeros(n, dtype=np.float32)

        u_row = self.user_to_row[user_id]
        sims = self.user_sim_matrix[u_row]
        top_nbrs = np.argsort(sims)[-self.top_k:]
        nbr_weights = sims[top_nbrs]
        weight_sum = np.sum(nbr_weights) + 1e-9

        if weight_sum <= 1e-6:
            return np.zeros(n, dtype=np.float32)

        nbr_interactions = self.interaction_matrix[top_nbrs].toarray()
        scores = np.zeros(n, dtype=np.float32)
        for i, aid in enumerate(candidate_article_ids):
            a_idx = self.pipe.article_id_to_idx.get(int(aid))
            if a_idx is not None:
                scores[i] = float(np.dot(nbr_weights, nbr_interactions[:, a_idx]) / weight_sum)
        return np.clip(scores, 0.0, 1.0)


class BaselineRecommenders:
    """Member 8: Standard Rubric Baselines (Random, Most-Popular, Pure Recency)."""

    def __init__(self, pipeline: DataEngineeringPipeline):
        self.pipe = pipeline
        self.rng = np.random.default_rng(123)

    def score_random(self, candidate_article_ids: list) -> np.ndarray:
        return self.rng.random(len(candidate_article_ids)).astype(np.float32)

    def score_most_popular(self, candidate_article_ids: list) -> np.ndarray:
        scores = np.zeros(len(candidate_article_ids), dtype=np.float32)
        for i, aid in enumerate(candidate_article_ids):
            idx = self.pipe.article_id_to_idx.get(int(aid))
            if idx is not None:
                scores[i] = float(self.pipe.articles_df.iloc[idx]["total_clicks"])
        return scores / (np.max(scores) + 1e-9)

    def score_recency(self, candidate_article_ids: list, current_time: pd.Timestamp) -> np.ndarray:
        scores = np.zeros(len(candidate_article_ids), dtype=np.float32)
        for i, aid in enumerate(candidate_article_ids):
            idx = self.pipe.article_id_to_idx.get(int(aid))
            if idx is not None:
                pub = self.pipe.articles_df.iloc[idx]["published_time"]
                age_h = max(0.0, (current_time - pub).total_seconds() / 3600.0)
                scores[i] = math.exp(-0.02 * min(age_h, 240.0))
        return scores


class HybridNewsRecommender:
    """
    Member 9: Hybrid Ensemble & Clean Prediction API Wrapper.
    Combines Content-Based, User-CF, Contextual, and Cold-Start switching logic,
    plus diversity-aware re-ranking (Member 12).
    """

    def __init__(
        self,
        pipeline: DataEngineeringPipeline,
        content_ctx: ContentAndContextualEngine,
        cf_model: UserBasedCollaborativeFilter,
        w_content: float = 0.45,
        w_cf: float = 0.30,
        w_context: float = 0.25,
        diversity_penalty: float = 0.12
    ):
        self.pipe = pipeline
        self.content_ctx = content_ctx
        self.cf_model = cf_model
        self.w_content = w_content
        self.w_cf = w_cf
        self.w_context = w_context
        self.diversity_penalty = diversity_penalty

    def predict_scores(
        self,
        user_id: int,
        candidate_article_ids: list,
        current_time: pd.Timestamp,
        user_location: str = "National"
    ) -> dict:
        """Returns component scores + unified hybrid score."""
        is_cold_user = user_id not in self.content_ctx.user_profiles

        s_content = self.content_ctx.score_content(user_id, candidate_article_ids)
        s_cf = self.cf_model.score_cf(user_id, candidate_article_ids)
        s_context = self.content_ctx.score_context(candidate_article_ids, current_time, user_location)
        s_trending = self.content_ctx.score_cold_start_trending(candidate_article_ids, current_time)

        if is_cold_user:
            s_hybrid = 0.65 * s_trending + 0.35 * s_context
        else:
            s_hybrid = (
                self.w_content * s_content +
                self.w_cf * s_cf +
                self.w_context * s_context
            )

        return {
            "hybrid": s_hybrid,
            "content": s_content,
            "cf": s_cf,
            "context": s_context,
            "trending": s_trending,
            "is_cold_user": is_cold_user,
        }

    def recommend(
        self,
        user_id: int,
        candidate_article_ids: list = None,
        current_time: pd.Timestamp = None,
        user_location: str = "National",
        top_k: int = 10,
        apply_diversity_rerank: bool = True
    ) -> pd.DataFrame:
        """Clean API endpoint returning ranked recommendations with explanations."""
        if current_time is None:
            current_time = self.pipe.articles_df["published_time"].max() + timedelta(hours=1)
        if candidate_article_ids is None:
            candidate_article_ids = self.pipe.articles_df["article_id"].tolist()

        score_dict = self.predict_scores(user_id, candidate_article_ids, current_time, user_location)
        scores = score_dict["hybrid"].copy()

        if apply_diversity_rerank and len(candidate_article_ids) > top_k:
            selected_indices = []
            seen_cats = defaultdict(int)
            remaining = list(range(len(candidate_article_ids)))
            art_df = self.pipe.articles_df

            for _ in range(min(top_k, len(remaining))):
                best_i = None
                best_adj_score = -1e9
                for idx_pos in remaining:
                    aid = int(candidate_article_ids[idx_pos])
                    a_idx = self.pipe.article_id_to_idx.get(aid)
                    cat = art_df.iloc[a_idx]["category_str"] if a_idx is not None else "Other"
                    adj = scores[idx_pos] - self.diversity_penalty * seen_cats[cat]
                    if adj > best_adj_score:
                        best_adj_score = adj
                        best_i = idx_pos
                selected_indices.append(best_i)
                remaining.remove(best_i)
                aid = int(candidate_article_ids[best_i])
                a_idx = self.pipe.article_id_to_idx.get(aid)
                if a_idx is not None:
                    seen_cats[art_df.iloc[a_idx]["category_str"]] += 1
            ranked_pos = selected_indices
        else:
            ranked_pos = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, pos in enumerate(ranked_pos, start=1):
            aid = int(candidate_article_ids[pos])
            a_idx = self.pipe.article_id_to_idx.get(aid)
            if a_idx is None:
                continue
            row = self.pipe.articles_df.iloc[a_idx]
            results.append({
                "rank": rank,
                "article_id": aid,
                "title": row["title"],
                "category": row["category_str"],
                "location": row.get("location", "National"),
                "published_time": row["published_time"],
                "hybrid_score": round(float(score_dict["hybrid"][pos]), 4),
                "content_sim": round(float(score_dict["content"][pos]), 4),
                "cf_score": round(float(score_dict["cf"][pos]), 4),
                "context_score": round(float(score_dict["context"][pos]), 4),
                "mode": "Cold-Start Fallback" if score_dict["is_cold_user"] else "Personalized Hybrid",
            })
        return pd.DataFrame(results)


# ==============================================================================
# TEAM 4: EVALUATION, TUNING & ETHICS AUDIT (Members 10, 11, 12)
# ==============================================================================

class EvaluationAndEthicsSuite:
    """
    Member 10: Offline Ranking Metrics (Precision@K, Recall@K, F1@K, NDCG@K, MRR)
    Member 11: Ablation Study & Weight Grid Search
    Member 12: Filter Bubble (diversity), Popularity Bias & Privacy Anonymizer
    """

    @staticmethod
    def ndcg_at_k(ranked_ids: list, ground_truth: set, k: int = 5) -> float:
        dcg = 0.0
        for i, aid in enumerate(ranked_ids[:k]):
            if aid in ground_truth:
                dcg += 1.0 / math.log2(i + 2)
        ideal_hits = min(len(ground_truth), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_models(
        self,
        eval_df: pd.DataFrame,
        hybrid_model: HybridNewsRecommender,
        baselines: BaselineRecommenders,
        k: int = 5,
        max_samples: int = 400
    ) -> pd.DataFrame:
        """Member 10 & 11: Evaluates all baselines, ablations, and final Hybrid model."""
        sample_df = eval_df.iloc[:max_samples]
        model_names = [
            "Random Baseline",
            "Pure Recency",
            "Most Popular",
            "User-Based CF",
            "Content-Based Only",
            "Content + Context",
            "Hybrid Ensemble (Full)",
        ]
        metrics = {m: {"p": [], "r": [], "ndcg": [], "mrr": []} for m in model_names}

        for _, row in sample_df.iterrows():
            uid = int(row["user_id"])
            cands = [int(x) for x in row["article_ids_inview"]]
            gt = set(int(x) for x in row["article_ids_clicked"])
            ts = row["impression_time"]
            loc = row.get("location", "National")
            if not cands or not gt:
                continue

            comp = hybrid_model.predict_scores(uid, cands, ts, loc)
            score_map = {
                "Random Baseline": baselines.score_random(cands),
                "Pure Recency": baselines.score_recency(cands, ts),
                "Most Popular": baselines.score_most_popular(cands),
                "User-Based CF": comp["cf"],
                "Content-Based Only": comp["content"],
                "Content + Context": 0.65 * comp["content"] + 0.35 * comp["context"],
                "Hybrid Ensemble (Full)": comp["hybrid"],
            }

            for m_name, scores in score_map.items():
                order = np.argsort(scores)[::-1]
                ranked_ids = [cands[idx] for idx in order]
                top_k_ids = ranked_ids[:k]
                hits = len(set(top_k_ids) & gt)

                metrics[m_name]["p"].append(hits / k)
                metrics[m_name]["r"].append(hits / len(gt))
                metrics[m_name]["ndcg"].append(self.ndcg_at_k(ranked_ids, gt, k))

                rr = 0.0
                for rank_i, aid in enumerate(ranked_ids, start=1):
                    if aid in gt:
                        rr = 1.0 / rank_i
                        break
                metrics[m_name]["mrr"].append(rr)

        summary = []
        for m_name in model_names:
            p_mean = float(np.mean(metrics[m_name]["p"]))
            r_mean = float(np.mean(metrics[m_name]["r"]))
            f1 = (2 * p_mean * r_mean / (p_mean + r_mean)) if (p_mean + r_mean) > 0 else 0.0
            summary.append({
                "Model": m_name,
                f"Precision@{k}": round(p_mean, 4),
                f"Recall@{k}": round(r_mean, 4),
                f"F1@{k}": round(f1, 4),
                f"NDCG@{k}": round(float(np.mean(metrics[m_name]["ndcg"])), 4),
                "MRR": round(float(np.mean(metrics[m_name]["mrr"])), 4),
            })
        return pd.DataFrame(summary)

    @staticmethod
    def run_ethics_and_privacy_audit(recs_df: pd.DataFrame, user_id: int) -> dict:
        """Member 12: Audits category diversity and hashes PII."""
        hashed_uid = hashlib.sha256(f"ebnerd_salt_{user_id}".encode()).hexdigest()[:16]
        unique_cats = recs_df["category"].nunique()
        diversity_ratio = unique_cats / max(1, len(recs_df))
        return {
            "anonymized_user_hash": hashed_uid,
            "unique_categories_surfaced": int(unique_cats),
            "category_diversity_ratio": round(diversity_ratio, 3),
            "filter_bubble_status": "Healthy Diversity" if diversity_ratio >= 0.4 else "High Concentration Warning",
        }


# ==============================================================================
# MAIN (offline evaluation run). The Streamlit UI lives in app.py.
# ==============================================================================

def main():
    print("=" * 75)
    print("RUNNING FULL HYBRID NEWS RECOMMENDATION PIPELINE")
    print("=" * 75)

    pipe = DataEngineeringPipeline(data_dir="./data")
    pipe.extract_and_load()
    pipe.build_nlp_representations(max_features=4000, svd_components=64)
    train_df, val_df, test_df, cold_df, cold_users = pipe.create_splits(cold_user_ratio=0.15)

    content_ctx = ContentAndContextualEngine(pipe, decay_lambda=0.015)
    content_ctx.build_user_profiles(train_df, cold_users)

    cf_model = UserBasedCollaborativeFilter(pipe, top_k_neighbors=25)
    cf_model.fit(train_df, cold_users)
    baselines = BaselineRecommenders(pipe)
    hybrid = HybridNewsRecommender(pipe, content_ctx, cf_model)

    evaluator = EvaluationAndEthicsSuite()
    results_df = evaluator.evaluate_models(test_df, hybrid, baselines, k=5, max_samples=300)
    print("\n[Members 10 & 11] OFFLINE EVALUATION & ABLATION TABLE (WARM TEST):")
    print(results_df.to_string(index=False))

    cold_results_df = evaluator.evaluate_models(cold_df, hybrid, baselines, k=5, max_samples=150)
    print("\n[Member 6 & 10] COLD-START HOLDOUT PERFORMANCE:")
    print(cold_results_df[cold_results_df["Model"].isin(
        ["Most Popular", "Pure Recency", "Hybrid Ensemble (Full)"])].to_string(index=False))

    sample_warm_user = next(iter(content_ctx.user_profiles.keys()))
    sample_recs = hybrid.recommend(user_id=sample_warm_user, top_k=5)
    audit = evaluator.run_ethics_and_privacy_audit(sample_recs, sample_warm_user)
    print(f"\n[Member 12] ETHICS & PRIVACY AUDIT: {audit}")


if __name__ == "__main__":
    main()
