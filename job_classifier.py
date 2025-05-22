# scraper.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import joblib
import os
from datetime import datetime

# Create results directory if missing
os.makedirs("results", exist_ok=True)

# 1. Scrape jobs from karkidi.com

def scrape_karkidi_jobs(keyword="data science", pages=1):
    # ... your scraping code ...
    service = Service(r'G:\WebDriver\chromedriver.exe')
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=service, options=options)
    
    for page in range(1, pages + 1):
        url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
        print(f"🔍 Scraping page {page} for keyword '{keyword}' ...")
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        job_blocks = soup.find_all("div", class_="ads-details")

        for job in job_blocks:
            try:
                title = job.find("h4").get_text(strip=True)
                company = job.find("a", href=lambda x: x and "Employer-Profile" in x).get_text(strip=True)
                location = job.find("p").get_text(strip=True)
                experience = job.find("p", class_="emp-exp").get_text(strip=True) if job.find("p", class_="emp-exp") else ""
                key_skills_tag = job.find("span", string="Key Skills")
                skills = key_skills_tag.find_next("p").get_text(strip=True) if key_skills_tag else ""
                summary_tag = job.find("span", string="Summary")
                summary = summary_tag.find_next("p").get_text(strip=True) if summary_tag else ""

                jobs_list.append({
                    "Title": title,
                    "Company": company,
                    "Location": location,
                    "Experience": experience,
                    "Summary": summary,
                    "Skills": skills
                })
            except Exception as e:
                print(f"⚠️ Error parsing a job: {e}")
                continue
        time.sleep(1)  # polite delay

   
    return pd.DataFrame(jobs_list)

# 2. Preprocess skills column and vectorize
def preprocess_skills(df):
    df = df.copy()
    df["Skills"] = df["Skills"].fillna("").str.lower()
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words="english")
    X = vectorizer.fit_transform(df["Skills"])
    return normalize(X), vectorizer

# 3. Cluster skills with KMeans
def cluster_skills(X, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)
    return model

# 4. Save clustering model & vectorizer to disk
def save_model(model, vectorizer, model_path="job_cluster_model.pkl", vec_path="tfidf_vectorizer.pkl"):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)

# 5. Load model and vectorizer
def load_model(model_path="job_cluster_model.pkl", vec_path="tfidf_vectorizer.pkl"):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer

# 6. Classify new job listings
def classify_new_jobs(new_df, model, vectorizer):
    new_df = new_df.copy()
    new_df["Skills"] = new_df["Skills"].fillna("").str.lower()
    X_new = vectorizer.transform(new_df["Skills"])
    new_df["Cluster"] = model.predict(normalize(X_new))
    return new_df

# 7. Notify users of jobs matching preferred clusters
def notify_user(new_df, user_preferred_clusters):
    matched_jobs = new_df[new_df["Cluster"].isin(user_preferred_clusters)]
    if not matched_jobs.empty:
        print("🚨 New jobs matching your preferred categories:")
        print(matched_jobs[["Title", "Company", "Location", "Skills", "Cluster"]])
    else:
        print("✅ No new jobs in your preferred categories at the moment.")
    return matched_jobs

# 8. Daily job check wrapper
def run_daily_job_check(keyword, user_clusters, model, vectorizer):
    print(f"\n🔄 Running daily job check for keyword: '{keyword}'")
    new_jobs = scrape_karkidi_jobs(keyword=keyword, pages=1)
    if new_jobs.empty:
        print("No jobs found for keyword.")
        return pd.DataFrame()
    
    classified = classify_new_jobs(new_jobs, model, vectorizer)
    matched = notify_user(classified, user_clusters)

    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    all_jobs_file = f"results/{keyword}_all_jobs_{timestamp}.csv"
    matched_jobs_file = f"results/{keyword}_matched_jobs_{timestamp}.csv"

    classified.to_csv(all_jobs_file, index=False)
    matched.to_csv(matched_jobs_file, index=False)

    print(f"📁 Saved all classified jobs: {all_jobs_file}")
    print(f"📁 Saved matched jobs: {matched_jobs_file}")

    return matched

# ----------- MAIN -------------
if __name__ == "__main__":
    # Step 1: Scrape initial data and save
    print("Starting initial scrape and clustering...")
    df_jobs = scrape_karkidi_jobs(keyword="data science", pages=2)
    if df_jobs.empty:
        print("No jobs scraped. Exiting.")
        exit(0)
    df_jobs.to_csv("results/karkidi_jobs.csv", index=False)

    # Step 2: Preprocess and cluster
    X_skills, tfidf_vectorizer = preprocess_skills(df_jobs)
    kmeans_model = cluster_skills(X_skills, n_clusters=5)
    save_model(kmeans_model, tfidf_vectorizer)

    # Step 3: Classify original jobs and save
    df_classified = classify_new_jobs(df_jobs, kmeans_model, tfidf_vectorizer)
    df_classified.to_csv("results/karkidi_classified_jobs.csv", index=False)

    # Step 4: Define user interests (clusters of interest)
    interests = {
        "data science": [0, 4],
        "cloud": [1],
        "machine learning": [2]
    }

    # Step 5: Run daily checks for each interest keyword
    for keyword, clusters in interests.items():
        run_daily_job_check(keyword, clusters, kmeans_model, tfidf_vectorizer)
