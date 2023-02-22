###--- Applying the K-Nearest Neighbors Algorithm built from scratch to the famous Titanic Dataset taken from Kaggle ----###

#Import the essential libraries--- Like Pandas for reading our dataset and converting it to a Dataframe.
import pandas as pd
import numpy as np
import random
import warnings
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

#Defined our train and test datasets by reading the csv files with the inbuilt function 'read_csv', and passing path of file as a parameter
data = pd.read_csv('C:\\Users\\Lenovo\\Documents\\ML_Project\\train.csv')

heatmap = sns.heatmap(data[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot = True)
plt.show()

print(data["SibSp"].unique())

bargraph_sibsp = sns.catplot(x = "SibSp", y = "Survived", data = data, kind = "bar")
plt.show()

age_visual = sns.FacetGrid(data,col = "Survived")
age_visual = age_visual.map(sns.distplot,"Age")
age_visual = age_visual.set_ylabels("Survival Probability")
plt.show()

sex_plot = sns.barplot(x = 'Sex',y = "Survived",data = data)
plt.show()
print(data[["Sex","Survived"]].groupby("Sex").mean())

print(data[["Pclass","Fare"]].groupby("Pclass").mean())
print(data[["Survived","Sex"]].groupby("Survived").count())

pclass = sns.catplot(x = "Pclass", y = "Survived", data = data, kind = "bar")
plt.show()

pclass_sex = sns.catplot(x = "Pclass", y = "Survived",hue = "Sex",data = data, kind = "bar")
plt.show()

embarked = sns.catplot(x = "Embarked", y = "Survived", data = data, kind = "bar")
plt.show()

print(data.head(10))

print(data.info())

mean = data["Age"].mean()
std = data['Age'].std()
is_null = data['Age'].isnull().sum()
print(is_null)

rand_age = np.random.randint(mean-std, mean+std, size = is_null)
age_slice = data["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age
data['Age'] = age_slice
is_null = data['Age'].isnull().sum()
print(is_null)

print(data.info())
data["Embarked"] = data["Embarked"].fillna("S")
print(data.info())
col_to_drop = ["PassengerId","Cabin","Name","Ticket","Fare"]
data.drop(col_to_drop, axis = 1 ,inplace = True)
print(data.head())

genders = {"male":0, "female":1}
data["Sex"] = data["Sex"].map(genders)
print(data.head(10))

ports = {"S":0,"C":1,"Q":2}
data["Embarked"] = data["Embarked"].map(ports)
print(data.head(10))

full_data = data.values.tolist()
print(full_data[:10])
random.shuffle(full_data)
print(full_data[:10])

test_size = 0.2
train_set = {0:[], 1:[]}
test_set  = {0:[], 1:[]}

train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[0]].append(i[1:])

for i in test_data:
    test_set[i[0]].append(i[1:])


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('k is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features-np.array(predict)))
            distances.append([euclidean_distance, group])
        #print(distances)
    #print(distances)
    distances = sorted(distances)
    #print(distances)
    votes = [i[1] for i in distances[:3]]
    #print(votes)
    #print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result



for group in test_set:
    correct = 0
    total = 0
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k = 5)
        if group == vote:
            correct+=1
        total+=1
    print("Accuracy:", correct/total)









