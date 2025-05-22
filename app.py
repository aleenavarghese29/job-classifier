# app.py

import streamlit as st
import pandas as pd
import joblib
from job_classifier import scrape_karkidi_jobs, classify_new_jobs, notify_user
from datetime import datetime
from sklearn.preprocessing import normalize

# Load model and vectorizer
model = joblib.load("job_cluster_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# App title
st.title("💼 Job Recommendation System")
st.write("This app scrapes jobs from Karkidi.com, clusters them by required skills, and notifies users based on their selected interests.")

# Sidebar - Select keyword and cluster interests
keyword = st.sidebar.selectbox("🔍 Choose job keyword", ["data science", "cloud", "machine learning", "AI", "NLP"])
user_clusters = st.sidebar.multiselect("🎯 Select preferred job clusters (0–4)", [0, 1, 2, 3, 4], default=[0, 2])

if st.sidebar.button("🚀 Run Job Check"):
    st.subheader(f"🔄 Checking new jobs for: {keyword}")
    new_jobs = scrape_karkidi_jobs(keyword=keyword, pages=1)
    classified = classify_new_jobs(new_jobs, model, vectorizer)
    matched = notify_user(classified, user_clusters)

    # Timestamp
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    all_jobs_path = f"results/{keyword}_all_jobs_{now}.csv"
    matched_jobs_path = f"results/{keyword}_matched_jobs_{now}.csv"

    classified.to_csv(all_jobs_path, index=False)
    matched.to_csv(matched_jobs_path, index=False)

    # Display and Download
    st.subheader("✅ Classified Jobs")
    st.dataframe(classified)
    st.download_button("⬇️ Download All Classified Jobs", data=classified.to_csv(index=False), file_name="classified_jobs.csv")

    st.subheader("🚨 Matched Jobs")
    st.dataframe(matched)
    if not matched.empty:
        st.download_button("⬇️ Download Matched Jobs", data=matched.to_csv(index=False), file_name="matched_jobs.csv")
    else:
        st.info("No matched jobs found today.")
else:
    st.info("Click the button in the sidebar to check new job listings.")
