#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[2]:


data = pd.read_csv(r"D:\DG- PG Studies\DG-Semester3\BDA3321_ML2\DG_NewProject\NewData\netflix_titles.csv")


# In[3]:


data.info()


# In[4]:


data.shape


# In[5]:


data.head()


# In[6]:


data.isna().sum()


# In[7]:


data['type'].value_counts()


# In[8]:


tv_show = data[data['type'] == 'TV Show']


# In[9]:


movies = data[data['type'] == 'Movie']


# In[10]:


tv_show.shape


# In[11]:


movies.shape


# In[12]:


tv_show.isna().sum()


# In[13]:


#Cleaning the TV Shows Dataset
tv_show.drop(['director'],axis=1,inplace=True)


# In[14]:


tv_show.drop(['country'],axis=1,inplace=True)


# In[15]:


tv_show.shape


# In[16]:


tv_show = tv_show.dropna(axis=0, how='any')


# In[17]:


tv_show.isna().sum()


# In[18]:


tv_show.tail(20)


# In[19]:


#Movies Data Cleaning
movies.isna().sum()


# In[20]:


movies.drop(['director','country'],axis=1,inplace=True)


# In[21]:


movies = movies.dropna(axis=0, how='any')


# In[22]:


movies.isna().sum()


# In[23]:


movies.tail(70)


# In[24]:


tv_show['rating'].value_counts()
#TV-MA: May be inappropriate for ages under 17. 
#TV-14: These shows may be unsuitable for children under 14
#TV-PG: Parental guidance is recommended; these programs may be unsuitable for younger children
#TV-Y7: Programs most appropriate for children age 7 and up.
#TV-Y: Programs aimed at a very young audience, including children from ages 2-6. 
#TV-G: Programs suitable for all ages
#NR: Not Rated
#R : Restricted, needs parental guidance for under 17


# In[25]:


movies['rating'].value_counts()
#PG-13: Parents Strongly Cautioned. Some Material May Be Inappropriate For Children Under 13
#PG: PG-rated content is not recommended for viewing by people under the age of 15 without guidance from parents. 
#G: Appropriate for people of all ages.
#NC-17: No One 17 and Under Admitted


# In[26]:


tv_show.drop(['date_added'],axis=1,inplace=True)
movies.drop(['date_added'],axis=1,inplace=True)


# In[27]:


tv_show.shape


# In[28]:


tv_show['id'] = range(1, len(tv_show) + 1)
tv_show.drop(['show_id'],axis=1,inplace=True)


# In[29]:


tv_show.head(20)


# In[30]:


movies['id'] = range(1, len(movies) + 1)
movies.drop(['show_id'],axis=1,inplace=True)
movies.head()


# In[31]:


import seaborn as sns


# In[32]:


#DO EDA for both 


# In[33]:


#Content-Based Filtering
tv_show.head(1)['description']


# In[34]:


#Based on pair-wise similarity
#We need to vectorize the tv_shows and movies


# In[35]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[36]:


tfv = TfidfVectorizer(min_df = 3, max_features = None, 
                     strip_accents = 'unicode', analyzer = 'word',
                      ngram_range = (1,3),stop_words = 'english')


# In[37]:


tv_show_m = tfv.fit_transform(tv_show['description'])


# In[38]:


tv_show_m
#Sparse matrix as most of the values will be zero. 


# In[39]:


tv_show_m.shape


# In[40]:


tv_show.shape


# In[41]:


movies.shape


# In[42]:


#Another method
from sklearn.metrics.pairwise import linear_kernel


# In[43]:


#Tv Shows
tv_coss = linear_kernel(tv_show_m,tv_show_m)


# In[44]:


tv_ind2 = pd.Series(tv_show['title'].index, index = tv_show['title']).drop_duplicates()


# In[45]:


tv_ind2


# In[46]:


tv_ind2['Chhota Bheem']


# In[47]:


def rec(title, cosine_sim = tv_coss):
    idx = tv_ind2[title]
    sim_score = enumerate(cosine_sim[idx])
    sim_score = sorted(sim_score,key=lambda x:x[1],reverse=True)
    sim_score = sim_score[1:11]
    sim_index = [i[0] for i in sim_score]
    print(tv_show['title'].iloc[sim_index])


# In[48]:


rec("Sex Education")


# In[49]:


#Movies 
movie_m = tfv.fit_transform(movies['description'])
movie_coss = linear_kernel(movie_m,movie_m)


# In[50]:


mov_ind = pd.Series(movies['title'].index, index = movies['title']).drop_duplicates()


# In[51]:


def mrec(title, cosine_sim = movie_coss):
    idx = mov_ind[title]
    sim_score = enumerate(cosine_sim[idx])
    sim_score = sorted(sim_score,key=lambda x:x[1],reverse=True)
    sim_score = sim_score[1:11]
    sim_index = [i[0] for i in sim_score]
    print(movies['title'].iloc[sim_index])


# In[52]:


mrec('Confessions of an Invisible Girl')


# In[53]:


#Collaborative Recommendation System - KNN 


# In[55]:


#KMeans Clustering
desc = movies['description'].values
type(desc)


# In[56]:


from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction import text


# In[57]:


stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

def tokenize(text):
    return [stemmer.stem(word) for word in tokenizer.tokenize(text.lower())]


# In[58]:


punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)


# In[59]:


vectorizer = TfidfVectorizer(stop_words = stop_words, tokenizer = tokenize)
X = vectorizer.fit_transform(desc)
word_features = vectorizer.get_feature_names()


# In[71]:


words = vectorizer.get_feature_names()


# In[60]:


kmeans = KMeans(n_clusters = 50, n_init = 5, n_jobs = -1)
kmeans.fit(X)


# In[64]:


common_words = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))


# In[66]:


Y = vectorizer.transform(["chrome browser to open."])
prediction = kmeans.predict(Y)
print(prediction)


# In[68]:


get_ipython().system('pip install wordcloud')


# In[74]:


from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
from PIL import Image
import requests


# In[77]:


comment_words = ''
stopwords = set(STOPWORDS)


# In[78]:


for val in data.description:  
    val = str(val) 
    tokens = val.split() 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
    comment_words += " ".join(tokens)+" "


# In[94]:


pic = np.array(Image.open(requests.get('http://www.clker.com/cliparts/O/i/x/Y/q/P/yellow-house-hi.png',stream=True).raw))
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, mask = pic, colormap = "gist_heat" ,
                min_font_size = 10).generate(comment_words)


# In[90]:


plt.figure(figsize = (10, 10), facecolor = 'black', edgecolor='red') 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()

