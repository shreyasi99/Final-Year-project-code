import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy import stats
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import pickle


def load_data(path):
    fn=pd.read_csv(path)
    return fn

def clean_data(file):
    file.drop(['StageID'], axis=1, inplace=True)
    file.PlaceofBirth = file.PlaceofBirth.replace(to_replace='KuwaIT', value='Kuwait')
    file.NationalITy = file.NationalITy.replace(to_replace='KW', value='Kuwait')
    file.columns = [x.capitalize() for x in file.columns]
    file.Nationality = [x.capitalize() for x in file.Nationality]
    file.Placeofbirth = [x.capitalize() for x in file.Placeofbirth]
    return file

edu_data=load_data(r"Datasets/xAPI-Edu-Data.csv")
edu_data=clean_data(edu_data)

#Vizualization
#Topic Distribution
sns.countplot(edu_data.Topic, palette='muted')


#Factor Plot Topic wise failed students
edu_data['Failed'] = np.where(edu_data['Class']=='L',1,0)
sns.factorplot('Topic','Failed', data=edu_data, size=7)

#Check for imbalanced data
sns.countplot(x='Class', data=edu_data)
plt.show()

#Parent school satisfaction
sns.countplot(x='Parentschoolsatisfaction', data = edu_data, palette='bright')
plt.show()

#Gaurdian relation effects in performance
sns.factorplot('Relation','Failed', data=edu_data)

#Gender of students effects in performance
sns.factorplot("Gender", "Failed", data=edu_data) #imp

#Hand raising effect on performance
fg=sns.FacetGrid(edu_data, hue='Failed', size=5, legend_out=False)
fg.map(sns.kdeplot, 'Raisedhands', shade=True)
fg.set(xlim=(0,edu_data['Raisedhands'].max()))
fg.add_legend()

#Discussion effect on performance
fg=sns.FacetGrid(edu_data, hue='Failed', size=5, legend_out=False)
fg.map(sns.kdeplot, 'Discussion', shade=True)
fg.set(xlim=(0,edu_data['Discussion'].max()))
fg.add_legend()

#Visited Resources effect on performance
fg=sns.FacetGrid(edu_data, hue='Failed', size=5, legend_out=False)
fg.map(sns.kdeplot, 'Visitedresources', shade=True) #imp
fg.set(xlim=(0,edu_data['Visitedresources'].max()))
fg.add_legend()

#Announcements Views effect on performance
fg=sns.FacetGrid(edu_data, hue='Failed', size=5, legend_out=False)
fg.map(sns.kdeplot, 'Announcementsview', shade=True)
fg.set(xlim=(0,edu_data['Announcementsview'].max()))
fg.add_legend()

edu_data.groupby('Topic').median()

edu_data['Absboolean'] = edu_data['Studentabsencedays']
edu_data['Absboolean'] = np.where(edu_data['Absboolean'] == 'Under-7',0,1)
edu_data['Absboolean'].groupby(edu_data['Topic']).mean()

###Preprocessing
def preprocessing(file):
    file.drop(['Absboolean','Failed'], axis=1, inplace=True)
    #Null values removal
    file.dropna(axis=0, how='any', inplace=True)
    #Seperating objects
    df_obj=file.select_dtypes(include=['object'])
    df_obj.drop(['Class'], axis=1, inplace=True)
    #Label encoding (for features)
    oe=OrdinalEncoder()
    df_new=oe.fit_transform(df_obj)
    df_new=pd.DataFrame(df_new, columns=df_obj.columns)
    #Combining
    file_new=file.copy()
    for col in file_new.columns.values:
        try:
            file_new[col]=df_new[col]
        except:
            pass
    return file_new

edu_adj=preprocessing(edu_data)
pd.plotting.radviz(edu_adj, 'Class')

###Splitting dataset
def split_data(file):
    X = file.iloc[:, :-1]
    y = file.Class
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(edu_adj)

le=LabelEncoder()
y_train2=le.fit_transform(y_train)
y_test2=le.transform(y_test)

###Clustering
##K-Means
kmeans=KMeans(n_clusters=3)
kmeans.fit(X_train)

y_test_pred=kmeans.predict(X_test)
acc_km=accuracy_score(y_test2, y_test_pred)
confusion_matrix(y_test2, y_test_pred)

#pickle.dump(kmeans, open('km_58', 'wb'))

#Cluster wise data (only contains test rows)
X_test_cl=pd.concat([X_test, pd.DataFrame(y_test2, index=X_test.index, columns=['y_test']), pd.DataFrame(y_test_pred, index=X_test.index, columns=['y_pred'])], axis=1)
df1=X_test_cl[X_test_cl.y_pred==0]
df2=X_test_cl[X_test_cl.y_pred==1]
df3=X_test_cl[X_test_cl.y_pred==2]

#Cluster wise data (only contains train rows)
y_train_pred=kmeans.predict(X_train)
X_train_cl=pd.concat([X_train, pd.DataFrame(y_train2, index=X_train.index, columns=['y_train']), pd.DataFrame(y_train_pred, index=X_train.index, columns=['y_pred'])], axis=1)
df1_train=X_train_cl[X_train_cl.y_pred==0]
df2_train=X_train_cl[X_train_cl.y_pred==1]
df3_train=X_train_cl[X_train_cl.y_pred==2]

#Mode of the clusters
m1=stats.mode(df1.y_test)
m2=stats.mode(df2.y_test)
m3=stats.mode(df3.y_test)

#Matching predicted clusters with labels of the target value assuming mode of the clusters to represent the cluster itself
y_test_pred_new=[]
for val in y_test_pred:
    if val==m1[0][0]:
        y_test_pred_new.append(0)
    elif val==m2[0][0]:
        y_test_pred_new.append(1)
    else:
        y_test_pred_new.append(2)

acc_km=accuracy_score(y_test2, y_test_pred_new)
confusion_matrix(y_test2, y_test_pred_new)

dict=pd.DataFrame({'Clusters':['Cluster1','Cluster2','Cluster3'],'Modal Class Value':le.inverse_transform([m1[0][0],m2[0][0],m3[0][0]])})
dict.to_csv('mode_values_kmeans.csv',index=False)

##Agglomerative Clustering
plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(X_test, method='ward'))
plt.axhline(y=280, color='r', linestyle='--')
plt.savefig('dendo.png')

agg = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
y_pred=agg.fit_predict(edu_adj.iloc[:,:-1])

plt.figure(figsize=(10, 7))
plt.scatter(edu_adj['Discussion'], edu_adj['Visitedresources'], c=agg.labels_)

y_true=le.transform(edu_adj.Class)
acc_agg=accuracy_score(y_true,y_pred)

#pickle.dump(agg, open('agg_48', 'wb'))

###Classification
##Naive Bayes
#On full dataset
gnb=GaussianNB()
gnb.fit(X_train, y_train2)
y_pred=gnb.predict(X_test)
acc_gnb=accuracy_score(y_test2, y_pred)

#pickle.dump(gnb, open('gnb_76', 'wb')) #0.7604166666666666

#On seperate clusters (clustered by kmeans)
#For cluster 1
gnb1=GaussianNB()
gnb1.fit(df1_train.iloc[:,:-2], df1_train.iloc[:,-2])
y_pred1=gnb1.predict(df1.iloc[:,:-2])
acc_1=accuracy_score(df1.iloc[:,-2],y_pred1)
#pickle.dump(gnb1, open('gnb_1_66', 'wb')) #0.66666666666666

#For cluster 2
gnb2=GaussianNB()
gnb2.fit(df2_train.iloc[:,:-2], df2_train.iloc[:,-2])
y_pred2=gnb2.predict(df2.iloc[:,:-2])
acc_2=accuracy_score(df2.iloc[:,-2],y_pred2)
#pickle.dump(gnb2, open('gnb_2_55', 'wb')) #0.5581395348837209

#For cluster 3
gnb3=GaussianNB()
gnb3.fit(df3_train.iloc[:,:-2], df3_train.iloc[:,-2])
y_pred3=gnb1.predict(df3.iloc[:,:-2])
acc_3=accuracy_score(df3.iloc[:,-2],y_pred3)
#pickle.dump(gnb3, open('gnb_3_86', 'wb')) #0.8695652173913043

#combined accuracy for cluster wise naive bayes model
acc_gnb_km=((acc_1*df1.shape[0])+(acc_2*df2.shape[0])+(acc_3*df3.shape[0]))/(df1.shape[0]+df2.shape[0]+df3.shape[0]) #0.6666666666666666 #0.6770833333333334 #0.71875

##Decision Tree
#On full dataset
#Tuning
dct=DecisionTreeClassifier()
params={
    'criterion':['gini','entropy'],
    'max_depth':[*range(1,25)],
    'min_samples_split':[*range(2,10)],
    'min_samples_leaf':[*range(1,5)]
}
grid=GridSearchCV(dct, params, n_jobs=4, scoring='accuracy', cv=5, verbose=1)
grid_res=grid.fit(X_train, y_train2)
g=grid_res.best_params_

dct=DecisionTreeClassifier(criterion=list(g.values())[0], max_depth=list(g.values())[1], min_samples_leaf=list(g.values())[2], min_samples_split=list(g.values())[3])
dct.fit(X_train, y_train2)
y_pred=dct.predict(X_test)
acc_dct=accuracy_score(y_test2, y_pred)
pickle.dump(dct, open('model/dct_76', 'wb'))

#dict=pd.DataFrame({'Model':['KMeans Clustering', 'Agglomerative Clustering', 'Naive Bayes', 'Naive Bayes + KMeans Clustering', 'Decision Tree'],'Accuracy':[acc_km, acc_agg, acc_gnb, acc_gnb_km, acc_dct]})
#dict.to_csv("all_accuracies.csv", index=False)












