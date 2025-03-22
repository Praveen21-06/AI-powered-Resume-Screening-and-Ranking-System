import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

st.set_page_config(page_title="AI Resume Screener", layout="wide")

st.markdown(
    """
    <style>
        body { background-color: #121212; color: #ffffff; }
        .stApp { background-color: #1e1e1e; padding: 20px; border-radius: 10px; }
        .stMarkdown { font-size: 18px; }
        .stTextArea textarea { background-color: #333; color: #ffffff; }
        .stFileUploader div { background-color: #333; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/912/912214.png", width=100)
st.sidebar.title("📌 Instructions")
st.sidebar.info(
    "1️⃣ Enter the **Job Description**.\n"
    "2️⃣ Upload **Multiple Resumes (PDF)**.\n"
    "3️⃣ AI will **Rank Candidates** based on relevance.\n"
    "4️⃣ View the **Sorted Results** in a Table."
)

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = "".join(page.extract_text() or "" for page in pdf.pages)
    return text

def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], vectors[1:]).flatten()

st.markdown("<h1 style='text-align: center; color: #ff9800;'>📄 AI Resume Screener</h1>", unsafe_allow_html=True)

st.subheader("📝 Job Description")
job_description = st.text_area("Enter the job description", height=150)

st.subheader("📂 Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.subheader("📊 Ranking Resumes")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    resumes = [extract_text_from_pdf(file) for file in uploaded_files]
    
    for i, file in enumerate(uploaded_files):
        progress_bar.progress((i + 1) / len(uploaded_files))
        status_text.text(f"Processing {file.name}...")
    
    time.sleep(1)
    progress_bar.empty()
    status_text.success("Processing Completed ✅")
    
    if any(resumes):
        scores = rank_resumes(job_description, resumes)
        results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
        results = results.sort_values(by="Score", ascending=False)
        st.dataframe(results.style.format({"Score": "{:.2f}"}).bar(subset=["Score"], color="#ff9800"))
        st.balloons()
        st.success("✅ Ranking Completed! Check the Table Above.")
    else:
        st.warning("⚠️ No text could be extracted from the uploaded resumes.")
