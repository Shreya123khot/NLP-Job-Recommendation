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


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main{
    background:#f8f9fc;
}


.title{
    text-align:center;
    font-size:50px;
    font-weight:800;
    background:linear-gradient(90deg,#667eea,#764ba2);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    margin-top:20px;
}


.subtitle{
    text-align:center;
    font-size:22px;
    color:#555;
    margin-bottom:40px;
}


.card{

    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.10);
    margin-top:20px;

}


.stButton button{

    width:100%;
    height:50px;
    border-radius:15px;
    background:linear-gradient(90deg,#667eea,#764ba2);
    color:white;
    font-size:18px;
    font-weight:bold;

}


.result{

    background:#e8f5e9;
    padding:30px;
    border-radius:20px;
    text-align:center;
    margin-top:25px;

}


.footer{

    text-align:center;
    color:#777;
    margin-top:50px;

}

</style>

""", unsafe_allow_html=True)



# ---------------- TITLE ----------------


st.markdown("""

<div class="title">
💼 AI Job Recommendation System
</div>

<div class="subtitle">

Find Your Best Career Opportunity Using NLP 🤖
<br>
Resume Based Intelligent Job Category Recommendation

</div>

""", unsafe_allow_html=True)



# ---------------- LOAD DATA ----------------


@st.cache_data
def load_data():

    df = pd.read_csv("Resume.csv")

    return df



df = load_data()



# ---------------- TEXT CLEANING ----------------


def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z ]',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()



# Dataset column check

if "Resume" in df.columns:

    resume_column = "Resume"

elif "Resume_str" in df.columns:

    resume_column = "Resume_str"

else:

    st.error("Resume column not found in dataset")

    st.stop()



df["Clean_Resume"] = df[resume_column].apply(clean_text)



# ---------------- NLP MODEL ----------------


@st.cache_resource
def train_model(data):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )


    vectors = vectorizer.fit_transform(
        data["Clean_Resume"]
    )


    return vectorizer, vectors



vectorizer, resume_vectors = train_model(df)



# ---------------- USER INPUT ----------------


st.markdown(
"""
<div class="card">

<h2>📄 Upload Your Resume Details</h2>

</div>
""",
unsafe_allow_html=True
)


user_resume = st.text_area(
    "Paste your resume skills, education and experience",
    height=200
)



# ---------------- BUTTON ----------------


if st.button("🚀 Recommend Job"):


    if user_resume.strip()=="":

        st.warning(
            "Please enter resume details"
        )


    else:


        cleaned = clean_text(
            user_resume
        )


        user_vector = vectorizer.transform(
            [cleaned]
        )


        similarity = cosine_similarity(
            user_vector,
            resume_vectors
        )


        index = similarity.argmax()


        category = df.iloc[index]["Category"]


        score = similarity[0][index]*100



        st.markdown(

        f"""

        <div class="result">

        <h2>🎯 Recommended Career Category</h2>

        <h1>{category}</h1>

        <h3>Resume Match Score : {score:.2f}%</h3>

        </div>

        """,

        unsafe_allow_html=True

        )



# ---------------- FOOTER ----------------


st.markdown(

"""
<div class="footer">

✨ Powered by NLP | TF-IDF | Cosine Similarity

</div>
""",

unsafe_allow_html=True

)
