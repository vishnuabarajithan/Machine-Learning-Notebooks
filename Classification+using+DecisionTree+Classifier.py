
# coding: utf-8

# <p style="font-family: Arial; font-size:2.75em;color:purple; font-style:bold">
# 
# Classification of Weather Data <br><br>
# using scikit-learn
# <br><br>
# </p>

# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# Daily Weather Data Analysis</p>
# 
# In this notebook, we will use scikit-learn to perform a decision tree based classification of weather data.

# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# 
# Importing the Necessary Libraries<br></p>

# In[19]:

import pandas as pd
from pandas import Series, DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import scale 

from sklearn.tree import DecisionTreeClassifier

import numpy as np
import scipy
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb


# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# 
# Creating a Pandas DataFrame from a CSV file<br></p>
# 

# In[20]:

cars = pd.read_csv('mtcars.csv')
cars.columns = ['car_names','mpg','cyl','disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'auto trans', 'gear', 'carburetor']
cars.head()


# In[73]:

#features(X)= mpg(miles per gallon) and Carburetor
#Label(y)= auto trans(Automatic Transmission)

cars_data = cars.ix[:,(5,11)].values

X = scale(cars_data)
y = cars.iloc[:,9].values
#X
#y


# In[74]:

sb.regplot(x='mpg', y='carburetor', data=cars, scatter=True)


# In[75]:

mpg = cars['mpg']
carburetor = cars['carburetor']

spearmanr_coefficient, p_value = spearmanr(mpg, carburetor)
spearmanr_coefficient, p_value


# In[76]:

cars.isnull().sum()


# In[77]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[78]:

#type(X_train)

#type(X_test)
#type(y_train)
#type(y_test)
#X_train
y_train


# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# 
# Fit on Train Set
# <br><br></p>
# 

# In[85]:

classifier = DecisionTreeClassifier(max_leaf_nodes=10)
classifier.fit(X_train, y_train)


# In[86]:

type(classifier)


# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# 
# Predict on Test Set 
# 
# <br><br></p>
# 

# In[87]:

predictions = classifier.predict(X_test)


# In[88]:

predictions[:10]


# In[89]:

y_test[:10]


# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# 
# Measure Accuracy of the Classifier
# <br><br></p>
# 

# In[90]:

accuracy_score(y_true = y_test, y_pred = predictions)

