import gensim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
import numpy as np


def replace_umlauts(text):
    """
    Replaces german umlauts and sharp s in given text.

    :param text: text as str
    :return: manipulated text as str
    """
    res = text
    res = res.replace('ä', 'ae')
    res = res.replace('ö', 'oe')
    res = res.replace('ü', 'ue')
    res = res.replace('Ä', 'Ae')
    res = res.replace('Ö', 'Oe')
    res = res.replace('Ü', 'Ue')
    res = res.replace('ß', 'ss')
    res = res.replace(',', '')
    res = res.replace('.', '')
    res = res.replace('!', '')
    res = res.replace('?', '')
    res = res.replace(':', '')
    res = res.replace(';', '')
    res = res.replace('(', '')
    res = res.replace(')', '')
    res = res.replace('&', '')
    return res

def cleaning_text(text):
    """
    - removes stop-words (die, der, ein,  )
    - replaces 'umlaute' (ä,ö,ü,Ä,Ö,Ü,ß)

    :param text: text as string
    :return: clean text as string
    """
    #import stop n_words
    stop_words = stopwords.words('german')

    cap_stop_words = []
    for word in stop_words:
        cap_word = word.capitalize()
        cap_stop_words.append(cap_word)

    stop_words.extend(cap_stop_words)

    #clean_text = [ [word for word in texts.lower().split() if word not in stoplist]
    #                for text in texts]
    text = [word for word in text.split() if word not in stop_words]

    # text = [word for word in text.lower().split() if word not in stop_words]    #lower()

    clean_text = []
    for word in text:
        clean_word = replace_umlauts(word)
        clean_text.append(clean_word)

    return clean_text


def convert_to_feature_vec(text, keyed_vec_model):
    """
    - clean text (remove stop-words and 'umlaute')
    - convert each word in clean text to a feature vector,
    stack them up to a 2d numpy array with shape(n_features,n_words)

    :param text: text as strings
    :return: 2d numpy array with shape(n_features,n_words)
    """
    clean_text = cleaning_text(text)

    #clean_text = ["Schweiz","Franken","Deutschland","Euro","Grossbritannien","britische_Pfund","Japan","Yen","Russland","Rubel","USA","US-Dollar","Kroatien","Kuna"]
    #feature_mat = [KeyedVectors[word] for word in clean_text]
    #vectors = [keyed_vec_model[word] for word in clean_text]
    vector = np.zeros((300,))
    i = 0
    for word in clean_text:
        try:
            vector += keyed_vec_model[word]
            i += 1
        except KeyError:
            try:
                vector += keyed_vec_model[word.capitalize()]
            except KeyError:
                continue

    if i != 0:
        vector /= i
    return vector
