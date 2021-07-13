from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.utils import secure_filename
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import pickle
import shutil

app=Flask(__name__)
app.config['UPLOAD_FOLDER']="D:\\YMTS0297\\IEEE\\Student Performance Prediction using ML\\Uploaded_data"
app.config['SECRET_KEY']='b0b4fbefdc48be27a6123605f02b6b86'

full_data=None; df_encoded=None
X_train=None; X_test=None; y_train=None; y_test=None
X=None; y=None
y_train2=None; y_test2=None
#KMeans Training clusters
df1=None; df2=None; df3=None
#KMeans Testing clusters
df1_train=None; df2_train=None; df3_train=None;
# #All Accuracies
# accuracy=[]

oe = OrdinalEncoder() #Contains features encoding
le = LabelEncoder() #Contains target encoding only

def clean_data(file):
    file.drop(['StageID'], axis=1, inplace=True)
    file.PlaceofBirth = file.PlaceofBirth.replace(to_replace='KuwaIT', value='Kuwait')
    file.NationalITy = file.NationalITy.replace(to_replace='KW', value='Kuwait')
    file.columns = [x.capitalize() for x in file.columns]
    file.Nationality = [x.capitalize() for x in file.Nationality]
    file.Placeofbirth = [x.capitalize() for x in file.Placeofbirth]
    return file

###Preprocessing
def preprocessing(file):
    #Null values removal
    file.dropna(axis=0, how='any', inplace=True)
    #Seperating objects
    df_obj=file.select_dtypes(include=['object'])
    df_obj.drop(['Class'], axis=1, inplace=True)
    #Label encoding (for features)
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

###Splitting dataset
def split_data(file):
    X = file.iloc[:, :-1]
    y = file.Class
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=4)
    print(X_train)
    return X_train, X_test, y_train, y_test

def kmeans_clustering():
    ##K-Means
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_train)

    y_test_pred = kmeans.predict(X_test)

    X_test_cl = pd.concat([X_test, pd.DataFrame(y_test2, index=X_test.index, columns=['y_test']),
                           pd.DataFrame(y_test_pred, index=X_test.index, columns=['y_pred'])], axis=1)
    global df1, df2, df3
    # Cluster 1
    df1 = X_test_cl[X_test_cl.y_pred == 0]
    # Cluster 2
    df2 = X_test_cl[X_test_cl.y_pred == 1]
    # Cluster 3
    df3 = X_test_cl[X_test_cl.y_pred == 2]

    global df1_train, df2_train, df3_train
    y_train_pred = kmeans.predict(X_train)
    X_train_cl = pd.concat([X_train, pd.DataFrame(y_train2, index=X_train.index, columns=['y_train']),
                            pd.DataFrame(y_train_pred, index=X_train.index, columns=['y_pred'])], axis=1)
    # Cluster 1
    df1_train = X_train_cl[X_train_cl.y_pred == 0]
    # Cluster 2
    df2_train = X_train_cl[X_train_cl.y_pred == 1]
    # Cluster 3
    df3_train = X_train_cl[X_train_cl.y_pred == 2]

    m1 = stats.mode(df1.y_test)
    m2 = stats.mode(df2.y_test)
    m3 = stats.mode(df3.y_test)
    mode_vals=[m1[0][0],m2[0][0],m3[0][0]]

    # Matching predicted clusters with labels of the target value assuming mode of the clusters to represent the cluster itself
    y_test_pred_new = []
    for val in y_test_pred:
        if val == m1[0][0]:
            y_test_pred_new.append(0)
        elif val == m2[0][0]:
            y_test_pred_new.append(1)
        else:
            y_test_pred_new.append(2)

    acc_km=accuracy_score(y_test2, y_test_pred_new)
    # global accuracy
    # accuracy.append(acc_km)
    flash("KMeans Clustering performed Successfully",'secondary')
    return acc_km, mode_vals

def agg_clustering():
    agg = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    y_pred = agg.fit_predict(df_encoded.iloc[:, :-1])
    y_true = le.transform(df_encoded.Class)
    acc_agg=accuracy_score(y_true, y_pred)
    flash("Agglomerative Clustering performed Successfully", 'secondary')
    # accuracy.append(acc_agg)
    return acc_agg

def naive_bayes_classifier():
    gnb = GaussianNB()
    gnb.fit(X_train, y_train2)
    y_pred = gnb.predict(X_test)
    acc_gnb = accuracy_score(y_test2, y_pred)
    flash("Naive Bayes model created Successfully", 'secondary')
    # accuracy.append(acc_gnb)
    return acc_gnb

def DT_classifier_untuned():
    dct = DecisionTreeClassifier()
    dct.fit(X_train, y_train2)
    y_pred = dct.predict(X_test)
    acc_dct = accuracy_score(y_test2, y_pred)
    flash("Decision Tree without tuning model created Successfully", 'secondary')
    # accuracy.append(acc_dct)
    return acc_dct


def DT_classifier_tuned():
    dct=pickle.load(open("model/dct_76","rb")) #Tuned model using GridSearchCV
    y_pred = dct.predict(X_test)
    acc_dct = accuracy_score(y_test2, y_pred)
    flash("Decision Tree with Hyperparameter tuning model created Successfully", 'secondary')
    # accuracy.append(acc_dct)
    return acc_dct

def Naive_bayes_kmeans():
    # Modelling On seperate clusters (clustered by kmeans)
    # For cluster 1
    gnb1 = GaussianNB()
    gnb1.fit(df1_train.iloc[:, :-2], df1_train.iloc[:, -2])
    y_pred1 = gnb1.predict(df1.iloc[:, :-2])
    acc_1 = accuracy_score(df1.iloc[:, -2], y_pred1)

    # For cluster 2
    gnb2 = GaussianNB()
    gnb2.fit(df2_train.iloc[:, :-2], df2_train.iloc[:, -2])
    y_pred2 = gnb2.predict(df2.iloc[:, :-2])
    acc_2 = accuracy_score(df2.iloc[:, -2], y_pred2)

    # For cluster 3
    gnb3 = GaussianNB()
    gnb3.fit(df3_train.iloc[:, :-2], df3_train.iloc[:, -2])
    y_pred3 = gnb1.predict(df3.iloc[:, :-2])
    acc_3 = accuracy_score(df3.iloc[:, -2], y_pred3)

    # combined accuracy for cluster wise naive bayes model
    acc_gnb_km = ((acc_1 * df1.shape[0]) + (acc_2 * df2.shape[0]) + (acc_3 * df3.shape[0])) / (df1.shape[0] + df2.shape[0] + df3.shape[0])
    flash("Naive Bayes + KMeans clustering combined model created Successfully", 'secondary')
    # accuracy.append(acc_gnb_km)
    # dict=pd.DataFrame({'Accuracy':accuracy})
    # dict.to_csv('run_time_acc.csv', index=False)
    return acc_gnb_km

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/load', methods=["POST","GET"])
def load():
    if request.method=="POST":
        myfile=request.files['filename']
        ext=os.path.splitext(myfile.filename)[1]
        if ext.lower() == ".csv":
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
            os.mkdir(app.config['UPLOAD_FOLDER'])
            myfile.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(myfile.filename)))
            flash('The data is loaded successfully','success')
            return render_template('load_dataset.html')
        else:
            flash('Please upload a CSV type document only','warning')
            return render_template('load_dataset.html')
    return render_template('load_dataset.html')

@app.route('/view')
def view():
    #dataset
    myfile=os.listdir(app.config['UPLOAD_FOLDER'])
    global full_data
    full_data=pd.read_csv(os.path.join(app.config["UPLOAD_FOLDER"],myfile[0]))
    full_data=clean_data(full_data)
    return render_template('view_dataset.html', col=full_data.columns.values, df=list(full_data.values.tolist()))

@app.route('/split', methods=['POST','GET'])
def split():
    if request.method=="POST":
        test_size=float(request.form['size'])
        test_size=test_size/100
        global df_encoded
        #preprocessing
        df_encoded=preprocessing(full_data)
        #split
        global X, y, X_train, X_test, y_train, y_test, y_train2, y_test2
        X = df_encoded.iloc[:, :-1]
        y = df_encoded.Class
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=2)
        y_train2 = le.fit_transform(y_train)
        y_test2 = le.transform(y_test)
        flash('The dataset is transformed and split successfully','success')
        return redirect(url_for('train_model'))
    return render_template('split_dataset.html')

@app.route('/train_model', methods=['GET','POST'])
def train_model():
    if request.method=="POST":
        model_no=int(request.form['algo'])
        if model_no==0:
            print("U have not selected any model")
        elif model_no==1:
            acc_km, mode_vals = kmeans_clustering()
            return render_template('train_model.html', mode_vals=le.inverse_transform(mode_vals), acc=acc_km, model=model_no)
        elif model_no==2:
            acc_agg=agg_clustering()
            return render_template('train_model.html', acc=acc_agg, model=model_no)
        elif model_no==3:
            acc_gnb=naive_bayes_classifier()
            return render_template('train_model.html', acc=acc_gnb, model=model_no)
        elif model_no==4:
            acc_dct=DT_classifier_untuned()
            return render_template('train_model.html', acc=acc_dct, model=model_no)
        elif model_no==5:
            acc_dct=DT_classifier_tuned()
            return render_template('train_model.html', acc=acc_dct, model=model_no)
        elif model_no==6:
            acc_gnb_km=Naive_bayes_kmeans()
            return render_template('train_model.html', acc=acc_gnb_km, model=model_no)
    return render_template('train_model.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method=='POST':
        #Accepts all values
        f1=request.form['f1']
        f2 = request.form['f2']
        f3 = request.form['f3']
        f4 = request.form['f4']
        f5 = request.form['f5']
        f6 = request.form['f6']
        f7 = request.form['f7']
        f8 = request.form['f8']
        f9 = float(request.form['f9'])
        f10 = float(request.form['f10'])
        f11 = float(request.form['f11'])
        f12 = float(request.form['f12'])
        f13 = request.form['f13']
        f14 = request.form['f14']
        f15 = request.form['f15']

        all_obj_vals=[f1,f2,f3,f4,f5,f6,f7,f8,f13,f14,f15]

        all_obj_vals=oe.transform([all_obj_vals])

        all_vals=[]
        for i in all_obj_vals[0][0:8]:
            all_vals.append(i)
        all_vals.extend([f9,f10,f11,f12])
        for i in all_obj_vals[0][-3:]:
            all_vals.append(i)

        #Model
        dct = pickle.load(open("model/dct_76", "rb"))
        pred=dct.predict([all_vals])
        pred=le.inverse_transform(pred)[0]
        return render_template('prediction.html', pred=pred)
    return render_template('prediction.html')

if __name__=='__main__':
    app.run(debug=True)