# Dataset link:
# https://drive.google.com/file/d/1-UjSQT84jjA6tbFrBPyB1Jp68itRu_Zz/view?usp=sharing

import streamlit as st
import pandas as pd
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Job Recommendation System",
    page_icon="💼",
    layout="wide"
)


# ---------------- CSS ----------------

st.markdown("""
<style>

body{
background:#f5f7fb;
}

.header{
background:linear-gradient(135deg,#667eea,#764ba2);
padding:35px;
border-radius:20px;
text-align:center;
color:white;
box-shadow:0 10px 30px rgba(0,0,0,0.2);
}


.header h1{
font-size:45px;
}


.card{
background:white;
padding:25px;
border-radius:20px;
box-shadow:0 5px 20px rgba(0,0,0,0.1);
margin-top:20px;
}


.stButton button{

background:linear-gradient(90deg,#667eea,#764ba2);
color:white;
width:100%;
height:50px;
border-radius:15px;
font-size:18px;

}

</style>

""",unsafe_allow_html=True)



# ---------------- HEADER ----------------


st.markdown("""

<div class="header">

<h1>💼 AI Job Recommendation System</h1>

<h3>
Find Your Best Career Category Using NLP
</h3>

<p>
❤️ Python | 🧠 NLP | 📊 Machine Learning | 🚀 Streamlit
</p>

</div>

""",unsafe_allow_html=True)



# ---------------- LOAD DATA ----------------


@st.cache_data
def load_data():

    df=pd.read_csv("resume_dataset.csv")

    return df



df=load_data()



# ---------------- TEXT CLEANING ----------------


def clean_text(text):

    text=str(text)

    text=text.lower()

    text=re.sub('[^a-zA-Z]',' ',text)

    text=re.sub('\s+',' ',text)

    return text



df["Clean_Resume"]=df["Resume"].apply(clean_text)



# ---------------- MODEL ----------------


vectorizer=TfidfVectorizer(
    stop_words="english",
    max_features=5000
)


resume_vectors=vectorizer.fit_transform(
    df["Clean_Resume"]
)



# ---------------- USER INPUT ----------------


st.markdown("""
<div class="card">

<h2>📄 Upload Resume</h2>

</div>
""",unsafe_allow_html=True)



resume_file=st.file_uploader(
    "Upload your resume (.txt)",
    type=["txt"]
)



if resume_file:


    resume_text=resume_file.read().decode(
        "utf-8"
    )


    cleaned_resume=clean_text(resume_text)



    if st.button("🔍 Find Best Job Category"):



        user_vector=vectorizer.transform(
            [cleaned_resume]
        )


        similarity=cosine_similarity(
            user_vector,
            resume_vectors
        )


        index=similarity.argmax()


        score=similarity[0][index]*100


        category=df.iloc[index]["Category"]



        st.markdown("""
        <div class="card">

        <h2>🎯 Recommended Job Category</h2>

        </div>
        """,unsafe_allow_html=True)



        st.success(category)



        st.metric(
            "Matching Score",
            f"{score:.2f}%"
        )



        st.progress(
            int(score)
        )



        st.balloons()



#
