
# coding: utf-8

# In[25]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[26]:

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


# In[27]:

import nltk


# In[28]:

nltk.download("stopwords")


# In[29]:

import string


# In[30]:

string.punctuation


# In[34]:

useless_words = nltk.corpus.stopwords.words("english")


# # Text Cleaning

# In[50]:

import re
import nltk

from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in useless_words]
    review = ' '.join(review)
    corpus.append(review)
    print(corpus)
    
    

  


# # Bag Of Words

# In[54]:

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer( max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
#X
#y


# In[39]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# # Using Naive Bayes Algorithm

# In[40]:

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[41]:

prediction = classifier.predict(X_test)


# In[43]:

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, prediction)


# # Confusion Matrix

# In[44]:

confusion_matrix


# # Accuracy of the Model

# In[59]:

from sklearn.metrics import accuracy_score
accuracy_score(y_true = y_test, y_pred = prediction)

