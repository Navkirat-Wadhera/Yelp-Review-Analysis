import numpy as np 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

yelp = pd.read_csv("yelp.csv")

yelp.describe()

yelp['text length']=yelp['text'].apply(len)

yelp.head()

g=sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length',edgecolor='black')

sns.boxplot(yelp,x='stars',y='text length',palette='rainbow')

sns.countplot(data=yelp,x='stars',palette='rainbow',edgecolor='black')

g=yelp[['stars','cool','useful','funny','text length']]
stars = g.groupby('stars').mean()
stars

stars.corr()

sns.heatmap(data=stars.corr(),cmap='coolwarm',annot=True)

yelp_class=yelp[(yelp['stars']==1) | (yelp['stars']==5)]
yelp_class.head()

X=yelp_class['text']

y=yelp_class['stars']

from sklearn.feature_extraction.text import CountVectorizer

count=CountVectorizer()

X=count.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.naive_bayes import MultinomialNB

nb=MultinomialNB()

nb.fit(X_train,y_train)

predictions=nb.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))

from sklearn.feature_extraction.text import TfidfTransformer

tfidf=TfidfTransformer()

from sklearn.pipeline import Pipeline

pipeline =Pipeline([
    ('count',CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('nb',MultinomialNB())
])

yelp_class.head()

X=yelp_class['text']
y=yelp_class['stars']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

pipeline.fit(X_train,y_train)

pipe_pred=pipeline.predict(X_test)

print(confusion_matrix(y_test,pipe_pred))

print(classification_report(y_test,pipe_pred))