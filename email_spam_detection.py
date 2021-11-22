#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 20:07:12 2021

@author: ayeshauzair
"""

# Build an email spam detection mechanism using the text classification 
# technique. Make a pickle file that would be used to test new mail ids. 
# Use the dataset from your mailbox and do not upload any of your mails 
# when you upload the solution.

import pandas as pd
import nltk
import pickle
import re
from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import pandas_profiling as pp


# Load Dataset
df_emails = pd.read_csv("spam_or_not_spam.csv")

# Pandas Profiling
# profile = pp.ProfileReport(df_emails)
# profile.to_file("Emails_EdA.html")

# Explore Dataset details
print("\n\n------------------------------------------------------------\n\n")
print(df_emails.head(5))
print(df_emails.shape)
print(df_emails.columns)
print(df_emails.info())

# Reading Spam emails to check content type
# print(df_emails[df_emails['label']==1])

# Checking duplicates and removing them
df_emails.drop_duplicates(inplace = True)

# Resetting Index as [0, 1, 2, .... ] instead of inconsistent indexes [1, 8, 9, ... ]
df_emails.reset_index(drop=True, inplace=True)

# There's only one null email so we replace it with an empty string 
df_emails['email'] = df_emails['email'].fillna(" ")

# Checking changes in Dataframe
print("\n\n------------------------------------------------------------\n\n")
print(df_emails.shape)
print(df_emails.info())
print(df_emails.head(5))

# Putting Emails in X and Spam Labels (0: No Spam, 1: Spam) in y
X = df_emails['email']
y = df_emails['label']

# Save Pickle Files for X and y
f = open("X.pkl", "wb")
pickle.dump(X, f)
f.close()

f = open("y.pkl", "wb")
pickle.dump(y, f)
f.close()

# Creating corpus with 2873 (Max length of X) features from emails and applying regex
corpus = []
for i in range(0,len(X)):
    corp = re.sub(r"\\n","",str(X[i]))
    corp = re.sub(r"\W"," ",corp)
    corp = re.sub(r'^br$', ' ', corp)
    corp = re.sub(r'\s+br\s+', ' ', corp)
    corp = re.sub(r'\s+[a-z]\s+', ' ', corp)
    corp = re.sub(r'^b\s+', '', corp)
    corp = re.sub(r'\s+', ' ', corp)
    corp = corp.lower()
    corpus.append(corp)
# print(corpus)

# Initializing vectorizer and fitting the corpus
vectorizer = CountVectorizer(max_features=len(X), stop_words=stopwords.words("english"))
X = vectorizer.fit_transform(corpus).toarray()
transform = TfidfTransformer()
X = transform.fit_transform(X).toarray()

# Applying Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=0)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Classification Report
cr = classification_report(y_test, y_pred)
print("\n\n------------------------------------------------------------\n\n")
print(cr)

# Saving the trained model in pickle file
f = open("model.pkl", "wb")
pickle.dump(model, f)
f.close()

# Saving the transformed vectorizer in pickle file
f = open("vectorizer.pkl", "wb")
pickle.dump(vectorizer, f)
f.close()

# ---------------- Sample Testing ----------------

sample = [" We Buy Homes 4 Cash Ayesha Shafqat Sell Your Home Fast — For Cash! An offer for your property is in the preparation phase. Close in less than 10 days with no closing costs. House on record: 515 Poplar Ct Gray, GA    Review Offers Now >>   If you would no longer like to receive communication from us, click here. PO Box 660675 #31403, Dallas, TX 75266-0675  This is an email advertisement This email was from: 124 Broadkill Rd, #456, Milton, DE 19968  Update Profile or Unsubscribe ."]

# Step 1: Load Vectorizer
text_numeric = pickle.load(open("vectorizer.pkl", "rb"))

# Step 2: Transform the sample using vectorizer
sample_transform = text_numeric.transform(sample).toarray()

# Step 3: Load Trained Model

clf = pickle.load(open("model.pkl", "rb"))

# Step 4: Perform prediction on transformed sample
spam_detection = clf.predict(sample_transform)

# Result:
print("\n\n------------------------------------------------------------\n\n")
if spam_detection == 0:
    print("Result: Not a spam email")
else:
    print("Result: Spam email")
