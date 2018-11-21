import pandas as pd
import numpy as np
import gensim
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib

import preprocessing

PATH = 'data/train_data.csv'
PATH_Y = 'data/train_labels.csv'
PATH_VAL = 'data/val_data.csv'
n_clusters = 4

df = pd.read_csv(PATH, sep='\t', header = None, names = ["Sentences"])
print(df.Sentences.head())

# get trained model
keyed_vec_model = gensim.models.KeyedVectors.load_word2vec_format("german.model", binary=True)

feature_vec_list = []
for message in df.Sentences:
    feature_vec = preprocessing.convert_to_feature_vec(message, keyed_vec_model)
    feature_vec_list.append(feature_vec)


df['feature_vec'] = feature_vec_list
X = np.concatenate(df['feature_vec'], axis=0)
X = np.reshape(X, (300, int(X.shape[0]/300)) , order = 'F')
X = np.transpose(X)
print(X.shape)
print("last X", X[:,-1])


# kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
# print(kmeans.labels_)
# df['label'] = pd.DataFrame(kmeans.labels_)

df_Y = pd.read_csv(PATH_Y, sep='\t', header = None, names = ["True_Labels"])
Y = df_Y.True_Labels.values
print(Y.shape)
svclassifier = LinearSVC()
svclassifier.fit(X, Y)


Y_pred = svclassifier.predict(X)
df['label'] = pd.DataFrame(Y_pred)
print("Classification report training set: ", classification_report(Y, Y_pred))

# safe the model to file
filename = 'svm_model.sav'
joblib.dump(svclassifier, filename)
exit()

# test on validation set
df_val = pd.read_csv(PATH_VAL, sep='\t', header = None, names = ["Sentences"])
feature_vec_list_val = []
for message in df_val.Sentences:
    feature_vec = preprocessing.convert_to_feature_vec(message, keyed_vec_model)
    feature_vec_list_val.append(feature_vec)

df_val['feature_vec'] = feature_vec_list_val
X_val = np.concatenate(df_val['feature_vec'], axis=0)
X_val = np.reshape(X_val, (300, int(X_val.shape[0]/300)) , order = 'F')
X_val = np.transpose(X_val)

Y_val_pred = svclassifier.predict(X_val)
df_val['label'] = pd.DataFrame(Y_val_pred)
#print("Classification report test set: ", classification_report(Y_val, Y_val_pred))

for i in range(5):
    print("Label ", i)
    np.sum((Y == i))
    print(df_val[df.label.isin([i])].Sentences)

for i in range(5):
    print("Label ", i)
    print(np.sum((Y == i)))
