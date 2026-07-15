import streamlit as st
import pandas as pd
import pyyaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Load Authentication Config ---
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# --- 2. Render Login/Signup Interface ---
tab1, tab2 = st.tabs(["🔑 Log In", "📝 Sign Up"])

with tab1:
    try:
        authenticator.login()
    except Exception as e:
        st.error(e)

with tab2:
    try:
        email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(pre_authorized=None, captcha=False)
        if email_of_registered_user:
            st.success('User registered successfully! Go to the Log In tab.')
            with open('config.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
    except Exception as e:
        st.error(e)

# --- 3. Protected Content Block ---
if st.session_state["authentication_status"]:
    
    with st.sidebar:
        st.write(f"Logged in as: **{st.session_state['name']}**")
        authenticator.logout('Log out', 'sidebar')
        st.write("---")

    # Helper function to read PDFs
    def extract_text_from_pdf(file_bytes):
        pdf_reader = PdfReader(file_bytes)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text

    # Loading the Kaggle Data
    @st.cache_data
    def load_data():
        df = pd.read_csv("internships.csv")
        # Combine title, company, and location for matching comparison
        df['search_text'] = df['internship_title'].fillna('') + " " + df['company_name'].fillna('') + " " + df['location'].fillna('')
        return df

    try:
        df_internships = load_data()
    except FileNotFoundError:
        st.error("Error: 'internships.csv' not found. Please put it in the same folder!")
        st.stop()

    st.title("🎯 AI Internship Matcher")
    st.write("Upload your resume, and our NLP engine will match you with the best internships!")

    uploaded_file = st.sidebar.file_uploader("Upload your Resume (PDF)", type=["pdf"])
    num_results = st.sidebar.slider("How many matches to show?", min_value=3, max_value=15, value=5)

    if uploaded_file is not None:
        st.subheader("🔄 Processing your resume...")
        resume_text = extract_text_from_pdf(uploaded_file)
        
        if len(resume_text.strip()) == 0:
            st.error("Could not extract text. Please try another PDF.")
        else:
            st.success("Resume text extracted successfully!")
            
            # NLP Matching
            corpus = [resume_text] + df_internships['search_text'].tolist()
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            resume_vector = tfidf_matrix[0]
            internship_vectors = tfidf_matrix[1:]
            similarity_scores = cosine_similarity(resume_vector, internship_vectors).flatten()
            
            df_results = df_internships.copy()
            df_results['Match Score'] = similarity_scores
            df_results = df_results.sort_values(by='Match Score', ascending=False).head(num_results)
            df_results['Match Score'] = (df_results['Match Score'] * 100).round(1).astype(str) + "%"
            
            st.subheader(f"🏆 Top {num_results} Recommended Internships for You:")
            display_cols = ['internship_title', 'company_name', 'location', 'Match Score']
            cols_to_show = [c for c in display_cols if c in df_results.columns]
            st.dataframe(df_results[cols_to_show], use_container_width=True, hide_index=True)
    else:
        st.info("💡 Upload your PDF resume in the sidebar to begin matching!")

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password or sign up for a new account.')
