# Dataset link:
# https://drive.google.com/file/d/1-UjSQT84jjA6tbFrBPyB1Jp68itRu_Zz/view?usp=sharing

import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI Job Recommendation System",
    page_icon="💼",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
.main{
    background-color:#f5f7fa;
}
.title{
    text-align:center;
    color:#0E76A8;
    font-size:40px;
    font-weight:bold;
}
.sub{
    text-align:center;
    color:gray;
    font-size:18px;
}
.stButton>button{
    background:#0E76A8;
    color:white;
    border-radius:10px;
    height:50px;
    width:100%;
    font-size:18px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
tfidf = pickle.load(open("tfidf.pkl", "rb"))
df = pickle.load(open("resume_data.pkl", "rb"))

# TF-IDF Matrix
tfidf_matrix = tfidf.transform(df["Clean_Resume"])

# -------------------- RECOMMEND FUNCTION --------------------
def recommend_jobs(resume):

    vector = tfidf.transform([resume])

    similarity = cosine_similarity(vector, tfidf_matrix)

    index = similarity.argsort()[0][-5:][::-1]

    result = df.iloc[index]

    score = similarity[0][index]

    return result, score

# -------------------- HEADER --------------------
st.markdown("<p class='title'>💼 AI Job Recommendation System</p>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Find the Best Job Category using NLP</p>", unsafe_allow_html=True)

st.sidebar.title("📌 Menu")
st.sidebar.info("Paste your resume and click Recommend.")

# -------------------- INPUT --------------------
resume = st.text_area(
    "📄 Paste Your Resume Here",
    height=300,
    placeholder="Paste your resume..."
)

# -------------------- BUTTON --------------------
if st.button("🚀 Recommend Job"):

    if resume.strip() == "":
        st.warning("Please enter your resume.")
    else:

        result, score = recommend_jobs(resume)

        st.success("Recommendation Completed")

        st.subheader("🎯 Top Recommended Job Categories")

        for i in range(len(result)):

            category = result.iloc[i]["Category"]

            percentage = round(score[i] * 100, 2)

            st.write(f"### {i+1}. {category}")

            st.progress(min(int(percentage), 100))

            st.write(f"**Match Score : {percentage}%**")

            st.markdown("---")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    "<center>Developed using ❤️ Python | NLP | Streamlit</center>",
    unsafe_allow_html=True
)
