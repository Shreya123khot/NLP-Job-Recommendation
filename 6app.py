# Dataset link:
# https://drive.google.com/file/d/1-UjSQT84jjA6tbFrBPyB1Jp68itRu_Zz/view?usp=sharing

import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Job Recommendation System",
    page_icon="💼",
    layout="wide"
)


# ---------------- CUSTOM STYLE ----------------

st.markdown("""
<style>

.main{
    background:#f5f7fb;
}


.header{

    background:linear-gradient(135deg,#667eea,#764ba2);
    padding:40px;
    border-radius:25px;
    text-align:center;
    color:white;
    margin-bottom:30px;

}


.header h1{

    font-size:45px;

}


.header h2{

    font-size:25px;

}


.card{

    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.12);
    margin:20px 0px;

}


.stButton button{

    width:100%;
    height:50px;
    border-radius:15px;
    background:linear-gradient(90deg,#667eea,#764ba2);
    color:white;
    font-size:18px;

}


.result{

    background:#e8f5e9;
    padding:20px;
    border-radius:15px;
    text-align:center;

}

</style>

""", unsafe_allow_html=True)



# ---------------- HEADER ----------------


st.markdown("""

<div class="header">

<h1>💼 AI Job Recommendation System</h1>

<h2>Find Your Best Career Category Using NLP</h2>

<p>
Get personalized job category recommendations based on your resume
</p>

</div>

""", unsafe_allow_html=True)



# ---------------- LOAD DATA ----------------


@st.cache_data
def load_dataset():

    df = pd.read_csv("Resume.csv")

    return df



df = load_dataset()



# ---------------- TEXT CLEANING ----------------


def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z]',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text



df["Clean_Resume"] = df["Resume"].apply(clean_text)



# ---------------- VECTOR CREATION ----------------


vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)


resume_vectors = vectorizer.fit_transform(
    df["Clean_Resume"]
)



# ---------------- RESUME UPLOAD ----------------


st.markdown("""
<div class="card">

<h2>📄 Upload Your Resume</h2>

<p>
Upload your resume file to get suitable job categories.
</p>

</div>

""", unsafe_allow_html=True)



uploaded_file = st.file_uploader(
    "Choose Resume File",
    type=["txt"]
)



# ---------------- PREDICTION ----------------


if uploaded_file:


    resume = uploaded_file.read().decode(
        "utf-8"
    )


    cleaned_resume = clean_text(resume)



    if st.button(
        "🔍 Find Recommendation"
    ):


        user_vector = vectorizer.transform(
            [cleaned_resume]
        )


        similarity = cosine_similarity(
            user_vector,
            resume_vectors
        )


        best_match = similarity.argmax()


        category = df.iloc[best_match]["Category"]


        score = similarity[0][best_match] * 100



        st.markdown("""
        <div class="card">

        <h2>🎯 Recommended Career Category</h2>

        </div>
        """,
        unsafe_allow_html=True)



        st.markdown(f"""

        <div class="result">

        <h2>{category}</h2>

        </div>

        """,
        unsafe_allow_html=True)



        st.write("")


        st.subheader("Matching Score")

        st.progress(
            int(score)
        )


        st.success(
            f"{score:.2f}% Match Found"
        )



        st.balloons()



# ---------------- FOOTER ----------------


st.markdown("""
<br>

<center>

Thank you for using our Job Recommendation System 💼

</center>

""",
unsafe_allow_html=True)
