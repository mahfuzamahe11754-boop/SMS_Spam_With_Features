SMS Spam Classification using SVM

A machine learning project that classifies SMS messages as spam or ham (not spam) using a Support Vector Machine (LinearSVC), TF-IDF text features, and handcrafted style features.

Dataset

SMS Spam Collection Dataset from Kaggle (UCI ML). It contains about 5,500 SMS messages labelled ham or spam.

Download spam.csv from Kaggle. The file is not included in this repository.

What the project does
Loads and checks the data: shape, missing values, duplicates, class balance.
Exploratory analysis: message length plots and word clouds for spam and ham.
Text cleaning: lowercase, remove URLs, emails, numbers and punctuation, remove stopwords, lemmatize.
Feature engineering: adds 11 handcrafted features taken from the raw message (see below).
Model: TF-IDF (unigrams and bigrams, 5000 features) combined with the engineered features, then a LinearSVC classifier.
Comparison: TF-IDF only vs TF-IDF + engineered features.
Tuning and validation: GridSearchCV on the SVM C value, plus 5-fold cross-validation.
Prediction function: predict_sms(text) returns Spam or Ham for any new message.
Engineered features

Cleaning removes style clues such as capital letters, numbers and !. These features are measured on the original message before cleaning:

Feature	Meaning
char_len	Number of characters
word_cnt	Number of words
digit_cnt	Number of digits
upper_ratio	Share of letters that are capitals
allcaps_words	Number of fully capitalised words (e.g. URGENT)
exclaim_cnt	Number of !
question_cnt	Number of ?
currency_cnt	Number of $, £, € signs
has_url	1 if the message has a link
has_phone	1 if the message has a long digit run (phone number or short code)
avg_word_len	Average word length
Results

Run the notebook to see the scores. The comparison table in the notebook shows accuracy, precision, recall and F1 for both models.

Model	Accuracy	Precision	Recall	F1
TF-IDF only	fill in	fill in	fill in	fill in
TF-IDF + engineered features	fill in	fill in	fill in	fill in
How to run
Google Colab (recommended)
Open SMS_Spam_SVM_with_features.ipynb in Google Colab.
Run all cells.
When asked, upload spam.csv.
Locally

The upload cell uses google.colab. To run locally, skip that cell and place spam.csv in the same folder as the notebook.

bash
pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn
jupyter notebook SMS_Spam_SVM_with_features.ipynb
Example
python
predict_sms("URGENT! Call 09061701461 NOW to claim your £1000 PRIZE!!!")  # Spam
predict_sms("Hi, are we meeting tomorrow at 3 PM?")                       # Ham
Tech stack

Python, pandas, NumPy, scikit-learn, NLTK, Matplotlib, Seaborn, WordCloud

Project files
SMS_Spam_SVM_with_features.ipynb: the full notebook
README.md: this file
