# Comment out this code if it is incompatible with your Jupyter notebook

import spacy.cli

spacy.cli.download("en_core_web_sm")

# In[ ]:


# Comment out this code if it is incompatible with your Jupyter notebook

nlp = spacy.load("en_core_web_sm")

# In[ ]:


import pandas as pd

# In[1]:


doc0 = nlp(t0)
doc1 = nlp(t1)
doc2 = nlp(t2)
doc3 = nlp(t3)
doc4 = nlp(t4)
doc5 = nlp(t5)
doc6 = nlp(t6)

# In[ ]:


from spacy.lang.en.stop_words import STOP_WORDS


# In[ ]:


def lemmatize(doc):
    return [
        token.lemma_ for token in doc
        if not token.is_punct and not token.is_space
           and (token.text == "US" or not token.lower_ in STOP_WORDS)
           and not token.tag_ == "POS"
    ]


# ## Exercise 2: TF-IDF
#
# 1. Write a function `tf` that receives a string and a spaCy `Doc` and returns the number of times the word appears in the `lemmatize`d `Doc`
# 2. Write a function `idf` that receives a string and a list of spaCy `Doc`s and returns _the inverse of_ the number of docs that contain the word
# 3. Write a function `tf_idf` that receives a string, a spaCy `Doc` and a list of spaCy `Doc`s and returns the product of `tf(t, d) · idf(t, D)`.
# 4. Write a function `all_lemmas` that receives a list of `Doc`s and returns a `set` of all available `lemma`s
# 5. Write a function `tf_idf_doc` that receives a `Doc` and a list of `Doc`s and returns a dictionary of `{lemma: TF-IDF value}`, corresponding to each the lemmas of all the available documents
# 6. Write a function `tf_idf_scores` that receives a list of `Doc`s and returns a `DataFrame` displaying the lemmas in the columns and the documents in the rows.
# 7. Visualize the TF-IDF, like this:
#
# ![TF-IDF](https://github.com/Juanlu001/bts-mbds-data-science-foundations/blob/master/sessions/img/tf-idf.png?raw=1)

# ##1. Write a function `tf` that receives a string and a spaCy `Doc` and returns the number of times the word appears in the `lemmatized` `Doc`
#
# Strategy: use the lemmatize function developed above to create list of all lemmas represented in a Doc.  Then use Counter function to count number of times each lemma appears in the list and put the results in dictionary.  Finally, look up the requested string in the dictionary to see number of times it appears.

# In[ ]:


from collections import Counter


# In[ ]:


def tf(string, Doc):
    elements = lemmatize(Doc)  # creates list of all lemmas in docs
    elem_count = Counter(
        elements)  # creates dictionary where keys = lemmas, and values = number of times it appears in Doc
    return elem_count[string]  # looks up number of times the string appears, according to the dictionary


# ##2. Write a function `idf` that receives a string and a list of spaCy `Doc`s and returns _the inverse of_ the number of docs that contain the word
#
# Stategy: for all documents in Docs, run `tf` function to see if string appears (i.e., the output of `tf` is greater than zero).  Keep track of number of times it appears overall with a counter variable that increases by one every time string appears in a document.  Then take the inverse of that counter variable after all the documents in Docs are checked.

# In[ ]:


Docs = [doc0, doc1, doc2, doc3, doc4, doc5,
        doc6]  # consolidated list of individual docs; this will be used as an input in the following functions


# In[ ]:


def idf(string, Docs):
    doc_count = 0  # initialize counter

    for Doc in Docs:
        in_Doc = tf(string, Doc)  # tf function outputs number of times string appears in a Doc
        if in_Doc > 0:  # if the string appears in Doc (i.e, the count is > 0)...
            doc_count += 1  # increase the counter by 1

    return 1 / doc_count  # final result is inverse of doc_count


# ###3. Write a function `tf_idf` that receives a string, a spaCy `Doc` and a list of spaCy `Doc`s and returns the product of `tf(t, d) · idf(t, D)`.

# In[ ]:


def tf_idf(string, Doc, Docs):
    return tf(string, Doc) * idf(string, Docs)


# ###4. Write a function `all_lemmas` that receives a list of `Doc`s and returns a `set` of all available `lemma`s
#
# Stategy: Create a list.  Then for all documents in Docs, run the lemmatize function to get list of lemmas for that individual document, and put those lemmas into the list (must do this one by one or else the list elements will be lists instead of strings, and the next step won't work as intended).  After all the documents are done, take the `set` of the list to get rid of repeating lemmas.

# In[ ]:


def all_lemmas(Docs):
    lemma_list = []  # Initialize the list which will contain all the lemmas

    for Doc in Docs:  # for each individual Doc
        doc_lemmas = lemmatize(Doc)  # get the lemmas as a list...
        [lemma_list.append(elem) for elem in doc_lemmas]  # and add each element of that list to lemma_list

    # print(lemma_list) #use for troubleshooting

    return set(
        lemma_list)  # lemma_list likely contains repeated values; set command will only return unique values from lemma_list, which will be the output of the function


# ###5. Write a function tf_idf_doc that receives a Doc and a list of Docs and returns a dictionary of {lemma: TF-IDF value}, corresponding to each the lemmas of all the available documents.
#
# Strategy: create dictionary with all lemmas in Docs as keys (from the `all_lemmas` function).  Then, for each key/lemma, define the value as the TF-IDF value given by the `tf_idf` function.

# In[ ]:


def tf_idf_doc(Doc, Docs):
    docs_lemmas = all_lemmas(Docs)  # gets list of all lemmas from Docs
    d = {}  # initialize dictionary d.  The following for loop will populate the dictionary with the lemmas as keys, and the lemmas' respective tf_idf as values
    for lemma in docs_lemmas:
        d[lemma] = tf_idf(lemma, Doc, Docs)

    return d


# ###6. Write a function tf_idf_scores that receives a list of Docs and returns a DataFrame displaying the lemmas in the columns and the documents in the rows.
#
# Strategy: Initialize dictionary `data`.  Define the keys of that dictionary with the lemmas from Docs outputted by `all_lemmas` function.  Then for each key/lemma, define the value as a list that is populated with the `tf_idf` output for all the documents in Docs.  Finally, convert the dictionary to a DataFrame and define the row indices as the document numbers.

# In[ ]:


def tf_idf_scores(Docs):
    docs_lemmas = all_lemmas(Docs)
    data = {}
    # now initialize dictionary keys with empty lists for values
    for lemma in docs_lemmas:
        data[lemma] = []

    # now fill in values for each Doc and each key (lemma)
    for Doc in Docs:
        for lemma in docs_lemmas:
            data[lemma].append(tf_idf(lemma, Doc, Docs))

    #  print(data) #for troubleshooting

    df = pd.DataFrame(data)
    df.index = ['doc0', 'doc1', 'doc2', 'doc3', 'doc4', 'doc5', 'doc6']
    #  print(df) #for troubleshooting
    return df


# In[ ]:


df_tf_idf = tf_idf_scores(Docs)
df_tf_idf


# Alternate method: create a list where each element is a dictionary of lemmas and TF-IDF scores for each document in Docs (from `tf_idf_doc` function).  Then turn this list into a DataFrame and define row indices as the document names.
#
# This creates the same result as above, with similar execution times.

# In[ ]:


def tf_idf_scores2(Docs):
    data = []
    for Doc in Docs:
        row = tf_idf_doc(Doc, Docs)
        data.append(row)

    df = pd.DataFrame(data)
    df.index = ['doc0', 'doc1', 'doc2', 'doc3', 'doc4', 'doc5', 'doc6']
    return df


# In[ ]:


df_tf_idf2 = tf_idf_scores2(Docs)
df_tf_idf2

# ###7. Visualize the TF-IDF, like this:
#
# ![TF-IDF](https://github.com/Juanlu001/bts-mbds-data-science-foundations/blob/master/sessions/img/tf-idf.png?raw=1)

# First step is to import matplotlib.pyplot and seaborn libraries.

