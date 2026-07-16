# Dataset link:
# https://drive.google.com/file/d/1-UjSQT84jjA6tbFrBPyB1Jp68itRu_Zz/view?usp=sharing


import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Job Recommendation System", page_icon="💼", layout="wide")

st.markdown("""
<style>
.main{background:#f8f9fc;}
.title{text-align:center;font-size:50px;font-weight:800;background:linear-gradient(90deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.subtitle{text-align:center;font-size:20px;color:#666;margin-bottom:30px;}
.card{background:white;padding:25px;border-radius:18px;box-shadow:0 8px 20px rgba(0,0,0,.1);}
.stButton>button{width:100%;height:50px;border-radius:12px;background:linear-gradient(90deg,#667eea,#764ba2);color:white;font-size:18px;font-weight:bold;}
.footer{margin-top:50px;padding:20px;border-radius:18px;background:linear-gradient(90deg,#667eea,#764ba2);color:white;text-align:center;font-size:20px;font-weight:600;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">💼 AI Job Recommendation System</div><div class="subtitle">Find Your Best Career Opportunity Using AI</div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("Resume.csv")

df=load_data()

def clean_text(text):
    text=str(text).lower()
    text=re.sub(r"[^a-zA-Z ]"," ",text)
    return re.sub(r"\s+"," ",text).strip()

resume_col="Resume" if "Resume" in df.columns else "Resume_str"
df["Clean_Resume"]=df[resume_col].apply(clean_text)

@st.cache_resource
def train(data):
    vec=TfidfVectorizer(stop_words="english",max_features=5000)
    mat=vec.fit_transform(data["Clean_Resume"])
    return vec,mat

vectorizer,resume_vectors=train(df)

st.markdown('<div class="card"><h2>📄 Paste Your Resume</h2></div>', unsafe_allow_html=True)
user_resume=st.text_area("Resume",height=220)

if st.button("🚀 Recommend Job"):
    if not user_resume.strip():
        st.warning("Please enter resume details.")
    else:
        u=vectorizer.transform([clean_text(user_resume)])
        sim=cosine_similarity(u,resume_vectors)[0]
        top=sim.argsort()[-3:][::-1]

        st.subheader("🏆 Top 3 Recommendations")
        for i,idx in enumerate(top,1):
            cat=df.iloc[idx]["Category"]
            score=float(sim[idx]*100)
            st.write(f"{i}. **{cat}**")
            st.progress(min(int(score),100))
            st.caption(f"Match Score: {score:.2f}%")

st.markdown("""
<div class="footer">
🌟 <b>Your Dream Career Starts Here!</b> 🚀<br><br>
Explore Opportunities • Build Your Future • Achieve Success 💼✨
</div>
""", unsafe_allow_html=True)
