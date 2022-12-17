import pickle
import streamlit as st

NBmodel = pickle.load(open("spam.pkl","rb"))
cvectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def main():
    st.title("Email Spam Classifier")
    st.caption("With 98% accuracy")
    message = st.text_input("Enter mail: ")
    if(st.button("Check")):
        data = [message]
        vector = cvectorizer.transform(data).toarray()
        result = NBmodel.predict(vector)
        prediction = result[0]
        if prediction==1:
            st.error("This mail is spam")
        else:
            st.success("This mail is not a spam")


main()
