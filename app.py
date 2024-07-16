import re
import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from PyPDF2 import PdfReader
from docx import Document

# Function to load a pickle file
def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading {file_path}: {str(e)}")
        return None

# Load the trained classifier
clf = load_pickle('clf.pkl')
if clf is None:
    st.stop()

# Load the trained TfidfVectorizer
tfidfd = load_pickle('tfidf.pkl')
if tfidfd is None:
    st.stop()

# Function to clean resume text
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # Remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # Remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # Remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # Remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # Remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # Remove extra whitespace
    return resumeText

# Function to extract name, email, and phone number
def extract_contact_info(resume_text):
    name_pattern = re.compile(r"Name:\s*(.*)")
    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    phone_pattern = re.compile(r"\+?\d[\d -]{8,12}\d")
    
    name_match = name_pattern.search(resume_text)
    email_match = email_pattern.search(resume_text)
    phone_match = phone_pattern.search(resume_text)
    
    name = name_match.group(1) if name_match else "Name not found"
    email = email_match.group(0) if email_match else "Email not found"
    phone = phone_match.group(0) if phone_match else "Phone number not found"
    
    return name, email, phone

# Function to read resume content based on file type
def read_resume(file):
    if file.type == "text/plain":
        return file.read().decode("utf-8")
    elif file.type == "application/pdf":
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    else:
        st.error("Unsupported file type.")
        return None

# Streamlit app
def main():
    st.title("Resume Screening Application")

    uploaded_file = st.file_uploader("Choose a resume file", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        # Read resume content based on file type
        resume_text = read_resume(uploaded_file)

        if resume_text:
            # Extract name, email, and phone number
            name, email, phone = extract_contact_info(resume_text)

            # Clean the input resume
            cleaned_resume = cleanResume(resume_text)

            # Transform the cleaned resume using the trained TfidfVectorizer
            input_features = tfidfd.transform([cleaned_resume])

            # Make the prediction using the loaded classifier
            prediction_id = clf.predict(input_features)[0]

            # Map category ID to category name
            category_mapping = {
                15: "Java Developer",
                23: "Testing",
                8: "DevOps Engineer",
                20: "Python Developer",
                24: "Web Designing",
                12: "HR",
                13: "Hadoop",
                3: "Blockchain",
                10: "ETL Developer",
                18: "Operations Manager",
                6: "Data Science",
                22: "Sales",
                16: "Mechanical Engineer",
                1: "Arts",
                7: "Database",
                11: "Electrical Engineering",
                14: "Health and fitness",
                19: "PMO",
                4: "Business Analyst",
                9: "DotNet Developer",
                2: "Automation Testing",
                17: "Network Security Engineer",
                21: "SAP Developer",
                5: "Civil Engineer",
                0: "Advocate",
            }

            category_name = category_mapping.get(prediction_id, "Unknown")

            # Display the extracted information
            st.write(f"**Name:** {name}")
            st.write(f"**Email:** {email}")
            st.write(f"**Phone:** {phone}")
            st.write(f"**Predicted Category:** {category_name}")

if __name__ == "__main__":
    main()
