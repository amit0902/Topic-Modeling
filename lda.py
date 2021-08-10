# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 20:01:00 2021

@author: Amit Kulkarni
"""

#Data
import sys
import re, numpy as np, pandas as pd
from pprint import pprint

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])
warnings.filterwarnings("ignore",category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

#Streamlit
import streamlit as st

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV

html_temp = """
<div style ="background-color:orange;padding:13px">
<h1 style ="color:black;text-align:center;">Group - 1 Batch : P-60 </h1>
</div>
"""
# this line allows us to display the front end aspects we have 
# defined in the above code
st.markdown(html_temp, unsafe_allow_html = True)
result =""

# giving the webpage a title
st.title("Topic Prediction")
st.header("This application helps you classify News Topic from any given article whether it is Political or Sports")
st.subheader("This model accompanies LDA (Latent Dirichlet Allocation) Library")
a = st.text_input("Enter your Text Data:","Type here...")
if(st.button('Submit')):
    result = a.title()
    st.success(result)

def sent_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)
        
# Convert to list
df = pd.read_csv(r"C:\Users\Amit Kulkarni\Desktop\Topic Modeling\Topic_Modeling\Politics_Sports_News_Cluster.csv")
data = df.Headlines.values.tolist()

data_words = list(sent_to_words(data))
print(data_words[:1])

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out1 = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out1.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out1

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only Noun, Adj, Verb, Adverb
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:2])

vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10,                        # minimum reqd occurences of a word 
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,             # max number of uniq words
                            )

data_vectorized = vectorizer.fit_transform(data_lemmatized)

# Materialize the sparse data
data_dense = data_vectorized.todense()

# Compute Sparsicity = Percentage of Non-Zero cells
print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")

# Build LDA Model
lda_model = LatentDirichletAllocation(n_components=4,               # Number of topics
                                      max_iter=10,               # Max learning iterations
                                      learning_method='online',   
                                      random_state=100,          # Random state
                                      batch_size=20,            # n docs in each learning iter
                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                      n_jobs = -1,               # Use all available CPUs
                                     )
lda_output = lda_model.fit_transform(data_vectorized)

print(lda_model)  # Model attributes

# Log Likelyhood: Higher the better
print("Log Likelihood: ", lda_model.score(data_vectorized))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(data_vectorized))

# See model parameters
pprint(lda_model.get_params())

# Define Search Param
search_params = {'n_components': [2,4,6,8,10,12,14,16,18,20], 'learning_decay': [.5, .7, .9]}

# Init the Model
lda = LatentDirichletAllocation()

# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)

# Do the Grid Search
model.fit(data_vectorized)

# Best Model
best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))

# Create Document - Topic Matrix
lda_output = best_lda_model.transform(data_vectorized)

# column names
topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]

# index names
docnames = ["Doc" + str(i) for i in range(len(data))]

# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic

df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
df_topic_distribution.columns = ['Topic Num', 'Num Documents']
df_topic_distribution

# Topic-Keyword Matrix
df_topic_keywords = pd.DataFrame(best_lda_model.components_)

# Assign Column and Index
df_topic_keywords.columns = vectorizer.get_feature_names()
df_topic_keywords.index = topicnames

# View
df_topic_keywords.head()

# Show top n keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)        

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords

# Define function to predict topic for a given text document.
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def predict_topic(text, nlp=nlp):
    global sent_to_words
    global lemmatization

    # Step 1: Clean with simple_preprocess
    mytext_2 = list(sent_to_words(text))

    # Step 2: Lemmatize
    mytext_3 = lemmatization(mytext_2, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Step 3: Vectorize transform
    mytext_4 = vectorizer.transform(mytext_3)

    # Step 4: LDA Transform
    topic_probability_scores = best_lda_model.transform(mytext_4)
    topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), :].values.tolist()
    return topic, topic_probability_scores
a = [a]
topic, topic_probability_scores = predict_topic(text = a)
b = topic_probability_scores
#c = topic_probability_scores[1]
st.subheader('Topic KeyWords:')
st.write(topic)

st.subheader('Topic Probability')
st.write(b)

st.subheader('Topic Identified:')
st.write('This is Political' if pd.Series(b[0][0]>0.5).item() else 'This is Sports')

# # Predict the topic
# mytext = ["This week in US politics: Biden takes on Facebook, cosies up to Fox News, to battle vaccine hesitancy"]
# topic, prob_scores = predict_topic(text = mytext)
# print(topic)
# prob_scores[0][0]
# print('Political' if prob_scores[0][0]>0.5 else 'Sports')
# # Construct the k-means clusters
# from sklearn.cluster import KMeans
# clusters = KMeans(n_clusters=15, random_state=100).fit_predict(lda_output)

# from sklearn.metrics.pairwise import euclidean_distances

# nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# def similar_documents(text, doc_topic_probs, documents = data, nlp=nlp, top_n=5, verbose=False):
#     topic, x  = predict_topic(text)
#     dists = euclidean_distances(x.reshape(1, -1), doc_topic_probs)[0]
#     doc_ids = np.argsort(dists)[:top_n]
#     if verbose:        
#         print("Topic KeyWords: ", topic)
#         print("Topic Prob Scores of text: ", np.round(x, 1))
#         print("Most Similar Doc's Probs:  ", np.round(doc_topic_probs[doc_ids], 1))
#     return doc_ids, np.take(documents, doc_ids)

# html_temp = """
# <div style ="background-color:orange;padding:13px">
# <h1 style ="color:black;text-align:center;">Group - 1 Batch - P60 </h1>
# </div>
# """
# # this line allows us to display the front end aspects we have 
# # defined in the above code
# st.markdown(html_temp, unsafe_allow_html = True)
# result =""

# a = st.subheader("Enter your Text Data:")

# # Get similar documents
# mytext = ["How Politics Changed in 30 Years of Reforms: CMs became powerful, women voted more, west & south marched ahead of north & east"]
# doc_ids, docs = similar_documents(text=mytext, doc_topic_probs=lda_output, documents = data, top_n=1, verbose=True)
# print('\n', docs[0][:500])

# st.subheader("This model accompanies LDA (Latent Dirichlet Allocation) Library")
# # giving the webpage a title
# st.title("Topic Prediction")
# st.header("This application helps you classify News Topic from any given article whether is Political or Sports")