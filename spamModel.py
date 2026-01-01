import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

data = pd.read_csv(r"C:\Users\ASUS\Desktop\HalkaSpamModel\spam.csv")

# preprocessing and cleaning

# print(data.shape)
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham','spam'],['Not Spam','Spam'])
# print(data.shape)

# checking for null values
# print(data.isnull().sum())

# print(data.head())

# train model
mess = data['Message']
cat = data['Category']

# spliting the data into train data and test data
(mess_train,mess_test,cat_train,cat_test) = train_test_split(mess, cat, test_size=0.2)

cv = CountVectorizer(stop_words='english')
feature = cv.fit_transform(mess_train)

# cearting model
model = MultinomialNB()
model.fit(feature, cat_train)

# test model
feature_test = cv.transform(mess_test)
print(model.score(feature_test, cat_test))

# predict data
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result[0]

st.header('Spam Detection')


# output = predict('Congratulations, you won a lottery')
input_mess = st.text_input('Enter Message Here')

if st.button('Validate'):
    output = predict(input_mess)
    st.write(output)




# python -m streamlit run spamModel.py