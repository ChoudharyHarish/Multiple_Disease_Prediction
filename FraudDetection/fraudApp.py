import re;
import nltk;
import joblib;
import numpy as np;
from nltk.corpus import stopwords;
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import PorterStemmer;
import streamlit as st;
from streamlit_option_menu import option_menu;

news_model = joblib.load('news_model.joblib');
spam_model = joblib.load('SpamModel.joblib');

news_vectorizer = joblib.load('tfidf_vectorizer.joblib');
spam_vectorizer = joblib.load("MailVectorizer.joblib");

news_input = "daniel j flynn flynn hillari clinton big woman campu breitbart"
spam_input = "GDSC IIITDMK is thrilled to announce its Team for the academic year 2023-2024. Please find the attached document.Hearty Congratulations to the whole Team. If you are not selected, it doesn’t mean that you have fewer skills or less experience, it just means that those who got selected fit the available space more efficiently. So don’t lose hope nor be sad. There’s always a next time and still, you can be a part of all our activities.We will come back soon with so many exciting events and workshopsn Best regards, Harish Choudhary GDSC Lead"


stemmer = PorterStemmer();


def convertText(content):
    stemmed = re.sub('[^a-zA-Z]',' ',content)
    stemmed = stemmed.lower()
    stemmed = stemmed.split()
    stemmed = [stemmer.stem(word) for word in stemmed if not word in stopwords.words('english')]
    stemmed = ' '.join(stemmed)
    return stemmed

def preprocess_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]

    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return ' '.join(words)

def predictFakeNews(text):
    x = convertText(text);
    x = np.array([x]);
    x = news_vectorizer.transform(x);
    x.reshape(1,-1);
    prediction = news_model.predict(x);
    if (prediction[0] == 0):
     return "Not a Fake News";
    else:
     return 'This is a Fake News';


def predictSpamMail(x):
   x = preprocess_text(x);
   x = np.array([x]);
   x = spam_vectorizer.transform(x);
   prediction = spam_model.predict(x)
   if (prediction[0] == 0):
     return 'Mail is Not spam';
   else:
     return 'Mail is spam';

def main():

    prediction = ''

    with st.sidebar:
    
      selected = option_menu('Fraud Detection System',
                          ['Spam Mail Detection',
                           'Fake News Detection',],
                          icons=['envelope','newspaper'],
                          default_index=0)
        


    if (selected == 'Spam Mail Detection'):
        st.title('Spam Mail Prediction using ML')
        
        text_input = st.text_area("Enter the email message:", height=360, key="text_input")
        st.markdown('<style>textarea{resize:vertical;}</style>', unsafe_allow_html=True)
        

        if st.button('Predict Spam Mail'):
            prediction = predictSpamMail(text_input)

        if len(prediction) == 0:
            st.warning('Fill the prompt to see the result');  
            return   
        if('Not' in prediction):    
            st.success(prediction)
        else :
            st.error(prediction)

    if (selected == 'Fake News Detection'):
        st.title('Fake News Prediction using ML')
        
        text_input = st.text_area("Enter the news here:", height=360, key="text_input")
        st.markdown('<style>textarea{resize:vertical;}</style>', unsafe_allow_html=True)
        

        if st.button('Fake News Detection'):
            prediction = predictFakeNews(text_input)

        if len(prediction) == 0:
            st.warning('Fill the prompt to see the result');  
            return   
        if('Not' in prediction):    
            st.success(prediction)
        else :
            st.error(prediction)
    
    

     


if __name__ == "__main__":
    main()
