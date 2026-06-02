#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install pandas numpy scikit-learn matplotlib seaborn nltk')


# In[11]:


import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[69]:


nltk.download('stopwords')


# In[70]:


df=pd.read_csv("D:\Sentiment-Analysis\IMDB Dataset.csv.zip")
df_small=df.sample(n=10000,random_state=42)
df_small.to_csv("imdb_dataset_small.csv",index=False)
print(df_small.shape)


# In[71]:


print(df.shape)
print(df.info())
print(df['sentiment'].value_counts())


# In[72]:


stop_words=set(stopwords.words('english'))
def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-zA-Z]','',text)
    words=text.split()
    words=[word for word in words if word not in stop_words]
    return " ".join(words)


# In[73]:


df['clean_text']=df['review'].apply(clean_text)


# In[74]:


df['sentiment']=df['sentiment'].map({'positive':1, 
                                    'negative':0
                                   })


# In[75]:


tfidf=TfidfVectorizer()
X=tfidf.fit_transform(df['clean_text'])
y=df['sentiment']


# In[76]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[77]:


model=LogisticRegression()


# In[78]:


model.fit(X_train,y_train)


# In[79]:


y_pred=model.predict(X_test)


# In[80]:


accuracy=accuracy_score(y_test,y_pred)
print(accuracy)


# In[81]:


print(classification_report(y_test,y_pred))


# In[82]:


cm=confusion_matrix(y_test,y_pred)
print(cm)


# In[83]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("confusion matrix")
plt.savefig("confusion_matrix.png")
plt.show()


# In[84]:


review=["this is an amazing movie"]
review_clean=[clean_text(text) for text in review]
review_vector=tfidf.transform(review_clean)
prediction=model.predict(review_vector)
print(prediction)


# In[87]:


import pickle
pickle.dump(model,open("sentiment_model.pkl","wb"))
pickle.dump(tfidf,open("tfidf_vectorizer.pkl","wb"))


# In[89]:


import zipfile
with zipfile.ZipFile('tfidf_vectorizer.zip','w',zipfile.ZIP_DEFLATED)as zipf:
    zipf.write('tfidf_vectorizer.pkl')
    print("compression completed")


# In[ ]:





# In[ ]:




