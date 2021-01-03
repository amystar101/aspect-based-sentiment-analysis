import pandas as pd;
import numpy as np
import process;
import feature_processing
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import sk_svm
import nltk


# nltk downloads
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# loading the dataset
print("Loading the dataset .......")
reviews_df = pd.read_csv("../dataset/AllProductReviews.csv");


# removing neutral reviews 
reviews_df = reviews_df[reviews_df["ReviewStar"] != 3]

# preprocessing the reviews
reviews_df = process.preprocess(reviews_df);

# word stemming with Part of speech tagging
reviews_df = process.word_stemming(reviews_df);

# spliting the dataset in training, cross validation set and testing set
train_df,test_df = train_test_split(reviews_df,test_size = 0.4,shuffle = True)

# extracting the features from the text data
aspects,values = feature_processing.feature_extraction(train_df)


# creating the feature vectors
feature_vectors,y = feature_processing.create_feature_vector(train_df,aspects,values)
feature_vectors_test,y_test = feature_processing.create_feature_vector(test_df,aspects,values)

# creating dic of indexing the aspect-values
mp = {}
cnt = 0
for lis in feature_vectors:
    for element in lis:
        if element not in mp.keys():
            mp[element] = cnt
            cnt += 1
 
for lis in feature_vectors_test:
    for element in lis:
        if element not in mp.keys():
            mp[element] = cnt
            cnt += 1


# removing the reviews with no aspects
cnt = 0
ind = 0
to_remove = []
for lis in feature_vectors:
    if lis == []:
        cnt += 1
        to_remove.append(ind)
    ind += 1
print("reviews with no aspects in training set = ",round((cnt/len(train_df))*100,2)," %")

# removing the no aspects entries
for i in range(0,len(to_remove)):
    feature_vectors.pop(to_remove[i]-i)
    y.pop(to_remove[i]-i)

cnt = 0
ind = 0
to_remove = []
for lis in feature_vectors_test:
    if lis == []:
        cnt += 1
        to_remove.append(ind)
    ind += 1
print("reviews with no aspects in test set = ",round((cnt/len(test_df))*100,2)," %")
for i in range(0,len(to_remove)):
    feature_vectors_test.pop(to_remove[i]-i)
    y_test.pop(to_remove[i]-i)



# traing with sklearn svm function
model = sk_svm.sk_svm(feature_vectors,y,mp)

# test performance of sklearn svm
print("Performance on test set")
sk_svm.performance(model,feature_vectors_test,y_test,mp)
