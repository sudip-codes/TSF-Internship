import streamlit as st
import pickle
import numpy as np

model=pickle.load(open("model.pkl","rb"))

def predict_percentage(hours):
    input=np.array([[hours]]).astype(np.float64)
    predection=model.predict(input)
    return float(predection)

def main():
    st.title(" SCORE APPLICATION ")
    hours=st.text_input("HOURS OF STUDY")

    if st.button("PREDICT"):

        output=predict_percentage(hours)
        st.success("The Predicted Percentage is {}".format(output))



if  __name__=="__main__":
    main()
