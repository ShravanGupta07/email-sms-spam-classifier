📧                                                             **Email/SMS Spam Classifier**

A simple web application that classifies text as "Spam" or "Not Spam" using a machine learning model. This app is built with Streamlit, a Python library for creating interactive web applications, and utilizes Natural Language Processing (NLP) techniques to preprocess text data.

🚀 **Features**

**Text Classification:** Enter an email or SMS message to classify it as "Spam" or "Not Spam".

**Text Preprocessing:**
Converts text to lowercase
Removes punctuation and stop words
Applies stemming

**Machine Learning:** Uses TF-IDF vectorization and a pre-trained classification model to make predictions.


📥 **Installation**

**Clone the repository:**

git clone https://github.com/yourusername/spam-classifier.git

cd spam-classifier

**Install the required packages:**


pip install -r requirements.txt

**Download necessary NLTK resources:**

import nltk

nltk.download('stopwords')

nltk.download('punkt')

Ensure that vectorizer.pkl and model.pkl files are in the project directory. 
These files are essential for TF-IDF transformation and spam classification.

🛠️ **Model and Vectorizer Generation**

Before running the Streamlit application, you need to generate the model.pkl and vectorizer.pkl files. 

**Follow these steps:**

**Run the Jupyter Notebook:**

Locate the provided Jupyter notebook in the project directory.

Execute all the cells in the notebook. This will train the spam classification model and save the necessary files.

**Generated Files:**

After running the notebook, you will find model.pkl and vectorizer.pkl in the same directory.

Ensure these files are in the project directory, as they are essential for the TF-IDF transformation and spam classification in the Streamlit application.

▶️ **Usage**

**Run the Streamlit app:**


streamlit run app.py

Input your message in the text area provided and click "Predict".

The app will display whether the message is classified as "Spam" or "Not Spam".

🧩**Code Overview**

transform_text(text)

This function preprocesses text to make it suitable for the model. 

**Steps include:**

Converting text to lowercase

Tokenizing

Removing punctuation and stop words

Stemming with PorterStemmer

TF-IDF Vectorizer

The app uses a pre-trained TF-IDF vectorizer (vectorizer.pkl) to convert text into a numerical format suitable for model input.

**Model Prediction** : 

The pre-trained classification model (model.pkl) predicts if the input text is "Spam" or "Not Spam".

💡 **Example**

Enter a message like:

"Congratulations! You've won a prize! Call now to claim your reward."

**Click Predict.**

If the message is spam, the app will display "Spam".

📋 **Requirements**

Python 3.x

Streamlit

nltk

pickle

Install necessary packages using:

pip install -r requirements.txt
