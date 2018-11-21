import pandas as pd
import numpy as np
import gensim
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib

import preprocessing

PATH = 'data/data.csv'
PATH_MODEL = 'svm_model.sav'
#load models
svclassifier = joblib.load(PATH_MODEL)
keyed_vec_model = gensim.models.KeyedVectors.load_word2vec_format("german.model", binary=True)

# test on validation set
df = pd.read_csv(PATH, sep='\t', header = None, names = ["Sentences"])
feature_vec_list = []
for message in df.Sentences:
    feature_vec = preprocessing.convert_to_feature_vec(message, keyed_vec_model)
    feature_vec_list.append(feature_vec)

df['feature_vec'] = feature_vec_list
X = np.concatenate(df['feature_vec'], axis=0)
X = np.reshape(X, (300, int(X.shape[0]/300)) , order = 'F')
X = np.transpose(X)

Y_pred = svclassifier.predict(X)
df['label'] = pd.DataFrame(Y_pred)
#print("Classification report test set: ", classification_report(Y, Y_pred))

for i in range(5):
    print("Label ", i)
    np.sum((Y_pred == i))
    print(df[df.label.isin([i])].Sentences)

for i in range(5):
    print("Label ", i)
    print("Number: ",np.sum((Y_pred == i)))
