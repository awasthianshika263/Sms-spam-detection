# SMS Spam Detection

This project is a simple machine learning based SMS spam classifier.  
It takes a text message as input and predicts whether the message is **Spam** or **Not Spam**.

I built this project to understand how text classification works using basic NLP techniques and to get hands-on experience with Python, scikit-learn, and Streamlit.

---

## How it works

- The dataset contains SMS messages labeled as `ham` or `spam`
- Messages are converted into numerical features using **CountVectorizer**
- A **Multinomial Naive Bayes** model is trained on the processed data
- The trained model predicts whether a new message is spam or not
- A simple Streamlit interface is used to test the model in real time

---

## Tech Stack

- Python  
- Pandas  
- scikit-learn  
- Streamlit  

---

## Dataset

The dataset used is a public SMS spam dataset containing labeled messages.  
Duplicate entries were removed during preprocessing.

---

## Model Performance

The model achieves around **97–99% accuracy** on the test data.  
(This may vary slightly due to random train-test splits.)

---

## How to run the project

1. Clone the repository  
   ```bash
   git clone <repository-url>
   
2. Navigate to the project folder  
   ```bash
   cd HalkaSpamModel
   
3. Install the required libraries
  pip install pandas scikit-learn streamlit

4. Run the Streamlit app
   streamlit run spamModel.py
   
6. Open the browser and test the model with your own SMS messages    
   

