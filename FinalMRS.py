#Libraries Required
import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#Creating an object
tfv = TfidfVectorizer(min_df = 3, max_features = None, 
                     strip_accents = 'unicode', analyzer = 'word',
                      ngram_range = (1,3),stop_words = 'english')

#Loading the Data 
data = pd.read_csv(r"C:\Users\Jerusha Gomez\OneDrive\Desktop\netflix_titles.csv",encoding='latin-1')
tv_show = data[data['type'] == 'TV Show']
movies = data[data['type'] == 'Movie']

#Data Cleaning of TV Shows Data 
tv_show.drop(['director','country'],axis=1,inplace=True)
tv_show = tv_show.dropna(axis=0, how='any')
tv_show['id'] = range(1, len(tv_show) + 1)
tv_show.drop(['show_id'],axis=1,inplace=True)
#st.dataframe(tv_show)

#Data Cleaning of Movies Data
movies.drop(['director','country'],axis=1,inplace=True)
movies = movies.dropna(axis=0, how='any')
movies['id'] = range(1, len(movies) + 1)
movies.drop(['show_id'],axis=1,inplace=True)
#st.dataframe(movies.head(10))

st.markdown("<h1 style='text-align: center; color: red;'>YOUR MARATHON & BINGE-WATCH FIX</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: white;'>Helping you choose your next set of movies or next set of TV shows to binge-watch!</h1>", unsafe_allow_html=True)
#Layout 
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Home','Movies','TV Shows','Age Appropriate','Main Themes','Data'])

with tab1:
    img1 = Image.open(r"C:\Users\Jerusha Gomez\Downloads\Poster.png")
    st.image(img1, caption="Representative Image of Movies and TV Shows")

    st.markdown(''' *Marathon* to movie-buffs means a 'movie marathon'. *Binge-Watch* is the term given to the phenomena of watching a 
    set of episodes (usually of TV shows). Unlike before, we now have movies and TV shows only a click away. While the limitations caused by
    having to go to a DVD store are almost done away with, we still find have one difficulty:  choosing a movie/ TV show to watch. This is where this
    website will prove useful!  

    *Basic Working:*    

    i.   Choose the tab based on what (Movie/TV Show) you want to watch.  
    ii.  From the drop-down list, pick an item that you have watched earlier.  
    iii. Based on the selected item, top ten recommendations are returned.  
    iv.  You can now look up these Movies/TV shows on your streaming sites.  
     v.   Grab some snacks, cozy up and begin your marathon/binge-watch!''')
    
with tab2:
    chosen = st.selectbox("Choose the movie you like:", movies['title'].unique())
    #Computing
    movie_m = tfv.fit_transform(movies['description'])
    movie_coss = linear_kernel(movie_m,movie_m)
    mov_ind = pd.Series(movies['title'].index, index = movies['title']).drop_duplicates()
    def mrec(title, cosine_sim = movie_coss):
        result = list()
        idx = mov_ind[title]
        sim_score = enumerate(cosine_sim[idx])
        sim_score = sorted(sim_score,key=lambda x:x[1],reverse=True)
        sim_score = sim_score[1:11]
        sim_index = [i[0] for i in sim_score]
        return sim_index
    button1 = st.button("Yep, I've watched this")
    if button1:
        ind = mrec(chosen)
        res = movies['title'].iloc[ind]
        res = res.reset_index()
        res.index = res.index+1
        st.table(res)

with tab3:
    chosen_tv = st.selectbox("Choose the TV show you like:", tv_show['title'].unique())
    #Computing
    tv_m = tfv.fit_transform(tv_show['description'])
    tv_coss = linear_kernel(tv_m,tv_m)
    tv_ind = pd.Series(tv_show['title'].index, index = tv_show['title']).drop_duplicates()
    def tvrec(title, cosine_sim = tv_coss):
        idx = tv_ind[title]
        sim_score = enumerate(cosine_sim[idx])
        sim_score = sorted(sim_score,key=lambda x:x[1],reverse=True)
        sim_score = sim_score[1:11]
        sim_index = [i[0] for i in sim_score]
        return sim_index
    button2 = st.button("Yes, I've watched this")
    if button2:
        ind_tv = tvrec(chosen_tv)
        res = tv_show['title'].iloc[ind_tv]
        res = res.reset_index()
        res.index = res.index+1
        st.table(res)
 
with tab4:
    age = st.radio("Age (in years) of the Intended Audience",("2-6","7-13","14-17","18+"))
    st.write("The Age Chosen is: ",age)
    if age=='2-6':
        cat = st.selectbox("Choose the Category",options=data['type'].unique())
        if cat=='Movie':
            movie26 = movies.loc[movies['rating'].isin(['G','TV-Y','TV-G'])]
            chosen26 = st.selectbox("Choose your favorite movie:", movie26['title'].unique())
            m26 = tfv.fit_transform(movie26['description'])
            mcoss26 = linear_kernel(m26,m26)
            mov_ind26 = pd.Series(movie26['title'].index, index = movie26['title']).drop_duplicates()
            def mrec26(title, cosine_sim = mcoss26):
                idx = mov_ind26[title]
                sim_score = enumerate(cosine_sim[idx])
                sim_score = sorted(sim_score,key=lambda x:x[1],reverse=True)
                sim_score = sim_score[1:11]
                sim_index = [i[0] for i in sim_score]
                return sim_index
            ind_2_6 = mrec26(chosen26)
            res = movie26['title'].iloc[ind_2_6]
            res = res.reset_index()
            res.index = res.index+1
            st.table(res)
        else:
            tv26 = tv_show.loc[tv_show['rating'].isin(['TV-Y','TV-G'])]
            chtv26 = st.selectbox("Choose your favorite TV show:", tv26['title'].unique())
            t26 = tfv.fit_transform(tv26['description'])
            tcoss26 = linear_kernel(t26,t26)
            tv_ind26 = pd.Series(tv26['title'].index, index = tv26['title']).drop_duplicates()
            def trec26(title, cosine_sim = tcoss26):
                idx = tv_ind26[title]
                sim_score = enumerate(cosine_sim[idx])
                sim_score = sorted(sim_score,key=lambda x:x[1],reverse=True)
                sim_score = sim_score[1:11]
                sim_index = [i[0] for i in sim_score]
                return sim_index
            tind_2_6 = trec26(chtv26)
            res = tv26['title'].iloc[tind_2_6]
            res = res.reset_index()
            res.index = res.index+1
            st.table(res)   
    elif(age=='7-13'):
        cat = st.selectbox("Choose the Category",options=data['type'].unique())
        if cat=='Movie':
            movie713 = movies.loc[movies['rating'].isin(['G','TV-Y','PG-13','TV-14','TV-Y7','TV-G'])]
            chosen713 = st.selectbox("Choose your favorite movie:", movie713['title'].unique())
            m713 = tfv.fit_transform(movie713['description'])
            mcoss713 = linear_kernel(m713,m713)
            mov_ind713 = pd.Series(movie713['title'].index, index = movie713['title']).drop_duplicates()
            def mrec713(title, cosine_sim = mcoss713):
                idx = mov_ind713[title]
                sim_score = enumerate(cosine_sim[idx])
                sim_score = sorted(sim_score,key=lambda x:x[1],reverse=True)
                sim_score = sim_score[1:11]
                sim_index = [i[0] for i in sim_score]
                return sim_index
            ind713 = mrec713(chosen713)
            res = movie713['title'].iloc[ind713]
            res = res.reset_index()
            res.index = res.index+1
            st.table(res)   
        else:
            tv713 = tv_show.loc[tv_show['rating'].isin(['TV-Y','TV-14','TV-Y7','TV-G'])]
            chtv713 = st.selectbox("Choose your favorite TV show:", tv713['title'].unique())
            t713 = tfv.fit_transform(tv713['description'])
            tcoss713 = linear_kernel(t713,t713)
            tv_ind713 = pd.Series(tv713['title'].index, index = tv713['title']).drop_duplicates()
            def trec713(title, cosine_sim = tcoss713):
                idx = tv_ind713[title]
                sim_score = enumerate(cosine_sim[idx])
                sim_score = sorted(sim_score,key=lambda x:x[1],reverse=True)
                sim_score = sim_score[1:11]
                sim_index = [i[0] for i in sim_score]
                return sim_index
            tind_713 = trec713(chtv713)
            res = tv713['title'].iloc[tind_713]
            res = res.reset_index()
            res.index = res.index+1
            st.table(res)  
    elif(age=='14-17'):
        cat = st.selectbox("Choose the Category",options=data['type'].unique())
        if cat=='Movie':
            movie1317 = movies.loc[movies['rating'].isin(['G','PG-13','TV-14','TV-Y7','TV-G','TV-MA','TV-PG'])]
            chosen1317 = st.selectbox("Choose your favorite movie:", movie1317['title'].unique())
            m1317 = tfv.fit_transform(movie1317['description'])
            mcoss1317 = linear_kernel(m1317,m1317)
            mov_ind1317 = pd.Series(movie1317['title'].index, index = movie1317['title']).drop_duplicates()
            def mrec1317(title, cosine_sim = mcoss1317):
                idx = mov_ind1317[title]
                sim_score = enumerate(cosine_sim[idx])
                sim_score = sorted(sim_score,key=lambda x:x[1],reverse=True)
                sim_score = sim_score[1:11]
                sim_index = [i[0] for i in sim_score]
                return sim_index
            ind1317 = mrec1317(chosen1317)
            res = movie1317['title'].iloc[ind1317]
            res = res.reset_index()
            res.index = res.index+1
            st.table(res)
        else:
            tv1317 = tv_show.loc[tv_show['rating'].isin(['TV-14','TV-Y7','TV-G','TV-MA','TV-PG'])]
            chtv1317 = st.selectbox("Choose your favorite TV show:", tv1317['title'].unique())
            t1317 = tfv.fit_transform(tv1317['description'])
            tcoss1317 = linear_kernel(t1317,t1317)
            tv_ind1317 = pd.Series(tv1317['title'].index, index = tv1317['title']).drop_duplicates()
            def trec1317(title, cosine_sim = tcoss1317):
                idx = tv_ind1317[title]
                sim_score = enumerate(cosine_sim[idx])
                sim_score = sorted(sim_score,key=lambda x:x[1],reverse=True)
                sim_score = sim_score[1:11]
                sim_index = [i[0] for i in sim_score]
                return sim_index
            tind_1317 = trec1317(chtv1317)
            res = tv1317['title'].iloc[tind_1317]
            res = res.reset_index()
            res.index = res.index+1
            st.table(res)
    else:
        st.write("Please refer to the Movies and TV Shows Tabs")

with tab5:
    img2 = Image.open(r"D:\DG- PG Studies\DG-Semester3\BDA3321_ML2\DG_NewProject\MovieRS\assets\download.png")
    st.image(img2, caption="Predominant Themes")

with tab6:
    st.markdown('''This is a website to recommend Movies and TV Shows - https://www.kaggle.com/datasets/shivamb/netflix-shows''')
    st.write("Movies Dataset Snippet:")  

    st.dataframe(movies.head(15))
    st.write("TV Shows Dataset Snippet:")  
    
    st.dataframe(tv_show.head(15))
    
