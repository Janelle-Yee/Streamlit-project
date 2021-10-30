## Setup environment, load the relevant libraries

import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from IPython.display import display
from streamlit_folium import folium_static
from PIL import Image
import folium
import csv
import missingno as msno

import seaborn as sns
sns.set(rc={'figure.figsize':(12,9)})
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency

import imblearn
from imblearn.over_sampling import SMOTE
from collections import defaultdict
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from kmodes.kmodes import KModes

import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from boruta import BorutaPy
from sklearn.feature_selection import RFECV
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

st.set_option('deprecation.showPyplotGlobalUse', False)

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.rc("font", size=12)

######################################################################################################################################################################################
html_temp = """<div style="background-color:lightblue; padding:1.5px">
<h1 style="color:while; text-align: center;">TDS 3301 Data Mining Project</h1>
</div><br>"""
st.markdown(html_temp, unsafe_allow_html=True)

#st.title("TDS 3301 Data Mining Project")
st.markdown("Group members: Yee Wen San (1181101326), Fiona Liou Shin Wee (1181100812), Denis Siow Chin Hsuen (118110466)")
st.subheader(" ")
st.header("QUESTION 1: Profiling Customers in a Self-Service Coin Laundry Shop")

#load data
data_load_state = st.text('Loading LaundryData...')
laundry = pd.read_csv("LaundryData.csv")
df = laundry.copy()
df
data_load_state.text('Loading Laundry_Data...done!')

#####################################################################################################################################################################################
st.text(" \n")
st.header('**Data Pre-processing**')
st.text(" \n")

# drop unwanted first column which is the index
df.drop(df.iloc[:,0:1], inplace=True, axis=1)

#show number of non-null values
st.markdown("Checking for missing values.")
df.isna().sum()
msno.bar(df)
st.pyplot()

st.text(" \n")
data_load_state = st.text('Filling null values with mode...')
st.text(" \n")

#Fill all missing values with mode 
for column in df.columns:
    df[column].fillna(df[column].mode()[0], inplace=True)
	
st.text(" \n")
data_load_state.text('Filling null values with mode...done!')
	
#show number of non-null values
st.markdown("Check again missing values")
df.isna().sum()
msno.bar(df)
st.pyplot()

st.text(" \n")
data_load_state = st.text('Undergoing pre-processing of data...')
st.text(" \n")

# change wrong character
df.Time = df.Time.replace(";", ":", regex=True)

# change datatype for Date and merge both Date and Time
df['Date'] = pd.to_datetime(df['Date']+ ' ' + df['Time'])
df.rename(columns = {'Date':'Timestamp'}, inplace = True)

# drop time column
df.drop(df.iloc[:,1:2], inplace=True, axis=1) 

# get hour, min, sec from timestamp
df['Hour'] = df.Timestamp.dt.hour
df['Minutes'] = df.Timestamp.dt.minute
df['Seconds'] = df.Timestamp.dt.second

df["Parts_Of_The_Day"] = pd.cut(df["Hour"], bins=[-1,1,6,12,17,21,24], labels=["Midnight", "Early Morning", "Morning", "Afternoon", "Evening", "Night"])

# normalize noisy data
#shirt_type
df.shirt_type = df.shirt_type.replace("long sleeve", "long_sleeve", regex=True)

#Pants_Colour
df.Pants_Colour = df.Pants_Colour.replace("blue_jeans", "blue", regex=True)
df.Pants_Colour = df.Pants_Colour.replace("blue  ", "blue", regex=True)
df.Pants_Colour = df.Pants_Colour.replace("blue ", "blue", regex=True)
df.Pants_Colour = df.Pants_Colour.replace("black ", "black", regex=True)

#Race
df.Race = df.Race.replace("foreigner ", "foreigner", regex=True)

# cut the 'Age_Range' into bins
df['Age_Range'] = pd.cut(df['Age_Range'], [0, 20, 30, 40, 50, 60, 70, 80, 90, 100], labels=['0-20', '20-30', '30-40', '40-50','50-60','60-70','70-80', '80-90','90-100'])

# rename column name shirt_type & pants_type
df.rename(columns={'shirt_type':'Shirt_type', 'pants_type': 'Pants_type'}, inplace = True)
st.text(" \n")
data_load_state.text('No missing values. Pre-processing data done!')

st.markdown("Dataframe after data pre-processing.")
st.write(df.astype('object'))

###############################################################################################################################################################################
# target variable Wash_Item (imbalanced data)
st.header("**Data Resampling**")
st.subheader("Dealing with imbalanced data using SMOTE")
st.text(" \n")
st.markdown("Data for Wash_Item is imbalanced.")
a = sns.countplot(x='Wash_Item', data = df)

for p in a.patches:
    a.annotate("%.0f" % p.get_height(), (p.get_x() + 
    p.get_width() / 2., p.get_height()), 
    ha='center', va='center', rotation=0, 
    xytext=(0, 18), textcoords='offset points')
st.pyplot()

########################################################################################################
df1 = df.copy()
df1 = df1.drop('Timestamp', axis = 1)

dictionary = defaultdict(LabelEncoder)
df1 = df1.apply(lambda x: dictionary[x.name].fit_transform(x))

y = df1.Wash_Item
X = df1.drop(columns = ['Wash_Item', 'Hour', 'Minutes', 'Seconds'] , axis=1) 

# perform SMOTE 
smt = imblearn.over_sampling.SMOTE(sampling_strategy="minority", random_state=10, k_neighbors=5)
X_res, y_res = smt.fit_resample(X, y)
colnames = X_res.columns

#######################################################################################################
st.text(" \n")
st.markdown("Dataframe after performing SMOTE")

dfsmote = pd.concat([X_res.reset_index(drop=True), y_res], axis=1) 
dfsmote = dfsmote.apply(lambda x: dictionary[x.name].inverse_transform(x))
dfsmote

#########################################################################################################
# target variable Wash_Item (balanced data)
st.text(" \n")
st.markdown("Data for Wash_Item is now balanced.")
b = sns.countplot(x='Wash_Item', data = dfsmote)

for p in b.patches:
    b.annotate("%.0f" % p.get_height(), (p.get_x() + 
    p.get_width() / 2., p.get_height()), 
    ha='center', va='center', rotation=0, 
    xytext=(0, 18), textcoords='offset points')
    #plt.savefig("balanced.png")
st.pyplot()

###############################################################################################################################################################################
# Which gender visits the laundry shop the most?
st.text(" \n")
st.header("**Exploratory Data Analysis (EDA)**")
st.subheader("Question 1: Which gender visits the laundry shop the most?")
st.text(" \n")

gender = sns.countplot(x='Gender', data = dfsmote)
for p in gender.patches:
    gender.annotate("%.0f" % p.get_height(), (p.get_x() + 
    p.get_width() / 2., p.get_height()), 
    ha='center', va='center', rotation=0, 
    xytext=(0, 18), textcoords='offset points')
st.pyplot()

#############################################################################
# Which race visits the laundry shop the most?
st.text(" \n")
st.subheader("Question 2: Which race visits the laundry shop the most?")
st.text(" \n")

race = sns.countplot(x='Race', data = dfsmote)
for p in race.patches:
    race.annotate("%.0f" % p.get_height(), (p.get_x() + 
    p.get_width() / 2., p.get_height()), 
    ha='center', va='center', rotation=0, 
    xytext=(0, 18), textcoords='offset points')
st.pyplot()

#############################################################################
# Which age_range visits the laundry shop the most?
st.text(" \n")
st.subheader("Question 3: Which age_range visits the laundry shop the most?")
st.text(" \n")

age = sns.countplot(x='Age_Range', data = dfsmote)
for p in age.patches:
    age.annotate("%.0f" % p.get_height(), (p.get_x() + 
    p.get_width() / 2., p.get_height()), 
    ha='center', va='center', rotation=0, 
    xytext=(0, 18), textcoords='offset points')
st.pyplot()

############################################################################
# Count the number of customers by gender and age_range
st.text(" \n")
st.subheader("Question 4: Count the number of customers by gender and age_range")
st.text(" \n")

g = sns.catplot(x="Gender", hue="Age_Range", kind="count", data=dfsmote)
ax = g.facet_axis(0,0)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
    (p.get_x() + p.get_width() / 2., p.get_height()), 
    ha = 'center', va = 'center', 
    xytext = (0, 10), textcoords = 'offset points')
st.pyplot()

############################################################################
# Which time has the most customers visiting the laundry shop?
st.text(" \n")
st.subheader("Question 5: Which time has the most customers visiting the laundry shop?")
st.text(" \n")

time = sns.countplot(x='Parts_Of_The_Day', data = dfsmote)
for p in time.patches:
    time.annotate("%.0f" % p.get_height(), (p.get_x() + 
    p.get_width() / 2., p.get_height()), 
    ha='center', va='center', rotation=0,
    xytext=(0, 20), textcoords='offset points')
st.pyplot()

############################################################################
# Count the number of customers by parts_of_the_day and attire
st.text(" \n")
st.subheader("Question 6: Count the number of customers by parts_of_the_day and attire")
st.text(" \n")

day_attire = sns.catplot(x="Parts_Of_The_Day", hue="Attire", kind="count", data=dfsmote)
ax = day_attire.facet_axis(0,0)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
    (p.get_x() + p.get_width() / 2., p.get_height()), 
    ha = 'center', va = 'center', 
    xytext = (0, 10), textcoords = 'offset points')
st.pyplot()

############################################################################
# Is there any relationship between basket size and race?
st.text(" \n")
st.subheader("Question 7: Is there any relationship between basket size and race?")
st.text(" \n")

# cross tabulation between Basket_Size and Race
CrosstabResult=pd.crosstab(index=dfsmote['Race'],columns=dfsmote['Basket_Size'])
CrosstabResult

# perform Chi-square test
chiSq = chi2_contingency(CrosstabResult)
st.text(" \n")

# p-value
st.text("P-Value of Chi-square Test:") 
chiSq[1]

#############################################################################
# What types of customers will likely to choose Washer No. 3 and Dryer No. 7? 
st.text(" \n")
st.subheader("Question 8: What types of customers will likely to choose Washer No. 3 and Dryer No. 7?")
st.text(" \n")

dfcompare = dfsmote.copy()
dfcompare.drop(dfcompare[dfcompare['Washer_No'] != 3].index, axis=0, inplace = True)
dfcompare.drop(dfcompare[dfcompare['Dryer_No'] != 7].index, axis=0, inplace = True)
dfcompare= dfcompare.reset_index(drop=True)
dfcompare = dfcompare.groupby(['Gender', 'Wash_Item']).sum()
dfcompare

#####################################################################################################################################################################################
st.text(" \n")
st.header("**Feature Selection**")

########################################################################
#  BORUTA
st.text(" \n")
st.subheader("BORUTA")

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

@st.cache(suppress_st_warning=True)
def boruta():
	rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5, random_state=1)
	feat_selector = BorutaPy(rf, n_estimators="auto", random_state=1)
	feat_selector.fit(X_res.values, y_res.values.ravel())
	
	boruta_score = ranking(list(map(float, feat_selector.ranking_)), colnames, order=-1)
	boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])

	boruta_score = boruta_score.sort_values("Score", ascending = False)

	bs_top = boruta_score.head(5)
	bs_bottom = boruta_score.tail(5)
	
	return bs_top, bs_bottom

bs_top, bs_bottom = boruta()
left_column, right_column = st.columns(2)
with left_column:
    st.text('Top 5 Results')
    bs_top

with right_column:
    st.text('Bottom 5 Results')
    bs_bottom

##########################################################################
# RFE
st.text(" \n")
st.subheader("RFE")

@st.cache(suppress_st_warning=True)
def rf():
	rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth = 5, n_estimators = 100, random_state=1)
	rf.fit(X_res,y_res)
	rfe = RFECV(rf, min_features_to_select = 1, cv = 3)
	rfe.fit(X_res,y_res)

	rfe_score = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
	rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
	rfe_score = rfe_score.sort_values("Score", ascending = False)

	rfe_top = rfe_score.head(5)
	rfe_bottom = rfe_score.tail(5)
	return rfe_top, rfe_bottom

rfe_top, rfe_bottom = rf()
left_column, right_column = st.columns(2)
with left_column:
    st.text('Top 5 Results')
    rfe_top

with right_column:
    st.text('Bottom 5 Results')
    rfe_bottom

#### Remove low ranking feature
X_res.drop(columns=["Spectacles"], axis=1, inplace=True)

######################################################################################################################################################################################

st.text(" \n")
st.header("**Association Rule Mining**")

df2 = df1.copy()
df2.drop(axis = 1, inplace = True, columns = ['Race', 'Gender', 'Age_Range', 'With_Kids', 'Kids_Category', 'Washer_No', 'Dryer_No', 'Wash_Item', 'Hour', 'Minutes', 'Seconds'])
df2 = df2.apply(lambda x: dictionary[x.name].inverse_transform(x))
df2_dummy = pd.get_dummies(df2)

arm_image = Image.open("figure16.png")
st.image(arm_image)
		 								
#####################################################################################################################################################################################
st.text(" \n")
st.header("**Machine Learning Techniques**")

#SVM
st.text(" \n")
st.subheader("Support Vector Machine (SVM)")

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3,random_state=1)

kernel_selection = st.radio('Choose desired kernel', ('linear', 'rbf', 'poly'))

if st.button('Generate SVM'):
	st.write("Kernel: " + str(kernel_selection))
	clf = svm.SVC(kernel=kernel_selection, gamma='auto', random_state=1, probability=True)

	#Train the model using the training sets
	clf.fit(X_train, y_train)

	#Predict the response for test dataset
	y_pred = clf.predict(X_test)

    # use model to predict probability that given y value is 1
	proba_SVM = clf.predict_proba(X_test)[::,1]

    # compute ROC curve and area the curve
	fpr_SVM, tpr_SVM, thresholds_SVM = roc_curve(y_test, proba_SVM)

	st.write("AUC: ", metrics.roc_auc_score(y_test, proba_SVM))
	st.write("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
	st.write("Precision: ",metrics.precision_score(y_test, y_pred))
	st.write("Recall: ",metrics.recall_score(y_test, y_pred))
	st.write("F1-score: ",metrics.f1_score(y_test, y_pred))

#####################################################################################################################################################################################
#KNN
st.text(" \n")
st.subheader("K-Nearest Neighbor (KNN)")

knn_range = st.slider('Choose the range of K', value=(1,5), min_value=1, max_value=10)
k_range = range(knn_range[0], knn_range[1]+1)

if st.button('Generate KNN'):
	scores = []
	for k in k_range:
		st.write("\n")
		st.write("Number of K: " + str(k))
		knn = KNeighborsClassifier(n_neighbors = k, weights='uniform')
		knn.fit(X_train, y_train)
		scores.append(knn.score(X_test, y_test))
		y_pred = knn.predict(X_test)
		proba_KNN = knn.predict_proba(X_test)[::,1]
		fpr_KNN, tpr_KNN, thresholds_KNN = roc_curve(y_test, proba_KNN)

		st.write("AUC: ", metrics.roc_auc_score(y_test, proba_KNN))
		st.write("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
		st.write("Precision: ",metrics.precision_score(y_test, y_pred))
		st.write("Recall: ",metrics.recall_score(y_test, y_pred))
		st.write("F1-score: ",metrics.f1_score(y_test, y_pred))

    
	plt.figure()
	plt.xlabel('k')
	plt.ylabel('accuracy')
	plt.title('Accuracy by n_neigbors')
	plt.scatter(k_range, scores)
	plt.plot(k_range, scores, color='green', linestyle='dashed', linewidth=1, markersize=5)
	st.pyplot()

#####################################################################################################################################################################################
#Logistic Regression
st.text(" \n")
st.subheader("Logistic Regression (LR)")

if st.button('Generate Logistic Regression'):
	logreg = LogisticRegression(solver='lbfgs', max_iter=200)
	logreg.fit(X_train, y_train)
	y_pred = logreg.predict(X_test)
	proba_LR = logreg.predict_proba(X_test)[::,1]
	fpr_LR, tpr_LR, thresholds_LR = roc_curve(y_test, proba_LR)

	st.write("AUC: ", metrics.roc_auc_score(y_test, proba_LR))
	st.write("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
	st.write("Precision: ",metrics.precision_score(y_test, y_pred))
	st.write("Recall: ",metrics.recall_score(y_test, y_pred))
	st.write("F1-score: ",metrics.f1_score(y_test, y_pred))

#####################################################################################################################################################################################
#Naive Bayes
st.text(" \n")
st.subheader("Naive Bayes (NB)")

if st.button('Generate Naive Bayes'):
	nb = GaussianNB()
	nb.fit(X_train, y_train)
	y_pred = nb.predict(X_test)
	proba_NB = nb.predict_proba(X_test)[::,1]
	fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test, proba_NB)

	st.write("AUC: ", metrics.roc_auc_score(y_test, proba_NB))
	st.write("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
	st.write("Precision: ",metrics.precision_score(y_test, y_pred))
	st.write("Recall: ",metrics.recall_score(y_test, y_pred))
	st.write("F1-score: ",metrics.f1_score(y_test, y_pred))

#####################################################################################################################################################################################
# Regression model
# Linear regression
st.text(" \n")
st.subheader("Linear regression")

if st.button('Generate Linear Regression'):
	regressor = LinearRegression()
	regressor.fit(X_train, y_train)
	y_pred = regressor.predict(X_test)

	result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
	result
	st.write("Mean Absolute Error: ",metrics.mean_absolute_error(y_test, y_pred))
	st.write("Mean Squared Error: ",metrics.mean_squared_error(y_test, y_pred))
	st.write("R-square score: ",metrics.r2_score(y_test, y_pred))

#####################################################################################################################################################################################
# Lasso Regression

st.text(" \n")
st.subheader("Lasso Regression")

if st.button('Generate Lasso Regression'):
	lassoM = linear_model.Lasso(alpha=1.0,normalize=True, max_iter=1e5)
	lassoM.fit(X_train, y_train)
	y_pred = lassoM.predict(X_test)

	result1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
	result1
	st.write("Mean Absolute Error: ",metrics.mean_absolute_error(y_test, y_pred))
	st.write("Mean Squared Error: ",metrics.mean_squared_error(y_test, y_pred))
	st.write("R-square score: ",metrics.r2_score(y_test, y_pred))

#####################################################################################################################################################################################
st.text(" \n")
st.header('**Classification Model Evaluation**')
st.text(" \n")

def evaluate():
	#SVM
	clf = svm.SVC(kernel='rbf', gamma='auto', random_state=1, probability=True)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	f1_SVM = metrics.f1_score(y_test, y_pred)
	proba_SVM = clf.predict_proba(X_test)[::,1]
	fpr_SVM, tpr_SVM, thresholds_SVM = roc_curve(y_test, proba_SVM)

	#KNN
	knn = KNeighborsClassifier(n_neighbors = 3, weights='uniform')
	knn.fit(X_train, y_train)
	y_pred = knn.predict(X_test)
	f1_KNN = metrics.f1_score(y_test, y_pred)
	proba_KNN = knn.predict_proba(X_test)[::,1]
	fpr_KNN, tpr_KNN, thresholds_KNN = roc_curve(y_test, proba_KNN)

	#LR
	logreg = LogisticRegression(solver='lbfgs', max_iter=200)
	logreg.fit(X_train, y_train)
	y_pred = logreg.predict(X_test)
	f1_LR = metrics.f1_score(y_test, y_pred)
	proba_LR = logreg.predict_proba(X_test)[::,1]
	fpr_LR, tpr_LR, thresholds_LR = roc_curve(y_test, proba_LR)

	#NB
	nb = GaussianNB()
	nb.fit(X_train, y_train)
	y_pred = nb.predict(X_test)
	f1_NB = metrics.f1_score(y_test, y_pred)
	proba_NB = nb.predict_proba(X_test)[::,1]
	fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test, proba_NB)

	scores = {'SVM': f1_SVM, 'KNN': f1_KNN, 'LR': f1_LR, 'Naive Bayes': f1_NB}
	model = list(scores.keys())
	score = list(scores.values())

	plt.figure(figsize=(7,6))
	plt.bar(model, score, width=0.5)
	for x,y in zip(model,score):

		label = "{:.2f}".format(y)

		plt.annotate(label,
					 (x,y), 
					 textcoords="offset points", 
					 xytext=(0,5),
					 ha='center')

	plt.title("F1-score of classification models")
	st.pyplot()

	plt.plot(fpr_SVM, tpr_SVM, color='red', label='SVM')
	plt.plot(fpr_KNN, tpr_KNN, color='blue', label='KNN')
	plt.plot(fpr_LR, tpr_LR, color='yellow', label='LR')
	plt.plot(fpr_NB, tpr_NB, color='purple', label='NB')

	plt.plot([0, 1], [0, 1], color='green', linestyle='--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic (ROC) Curve')
	plt.legend()
	#st.pyplot()

if st.button('Evaluate'):
	evaluate()
	image = Image.open("figure19.png")
	st.image(image)

#####################################################################################################################################################################################
st.text(" \n")
st.header("**KMode Clustering**")
st.text(" \n")

df3 = pd.concat([X_res.reset_index(drop=True), y_res], axis=1)

kmoderange = st.slider('Choose the range of K for K Mode clustering', value=(1,4), min_value=1, max_value=10)
kmode_range = range(kmoderange[0], kmoderange[1]+1)
range1 = range(kmoderange[0], kmoderange[1]+1, 1)

if st.button('Simulate'):
	distortions = []
	for num_clusters in kmode_range:
		kmode = KModes(n_clusters=num_clusters, init = "huang", n_init = 1, verbose=1)
		kmode.fit_predict(df3)
		distortions.append(kmode.cost_)

	y = np.array([i for i in range1])
	plt.plot(y, distortions, marker='o')
	plt.xlabel('K')
	plt.ylabel('Distortions')
	st.pyplot()

kmodecluster = st.number_input('Please choose the number of cluster k', min_value = 1, max_value = 10)

###################################################################################################################
if st.button('Simulate Clustering'):
	km = KModes(n_clusters=kmodecluster, init = "huang", n_init = 1, verbose=1)
	clusters = km.fit_predict(df3)

	df3 = df3.apply(lambda x: dictionary[x.name].inverse_transform(x))
	
	clusters_df = pd.DataFrame(clusters)
	clusters_df.columns = ['Cluster']
	df4 = pd.concat([df3, clusters_df], axis = 1).reset_index()
	df4 = df4.drop(df4.columns[0],axis=1)
	
	st.markdown('Cluster')
	c = sns.countplot(x='Cluster', data = df4)

	for p in c.patches:
		c.annotate("%.0f" % p.get_height(), (p.get_x() + 
		p.get_width() / 2., p.get_height()), 
		ha='center', va='center', rotation=0, 
		xytext=(0, 18), textcoords='offset points')
	st.pyplot()

	st.markdown("Attire")
	plt.subplots(figsize = (15,5))
	sns.countplot(x=df4['Attire'], order=df4['Attire'].value_counts().index, hue=df4['Cluster'])
	st.pyplot()

	st.markdown('Age Range')
	plt.subplots(figsize = (15,5))
	sns.countplot(x=df4['Age_Range'], order=df4['Age_Range'].value_counts().index, hue=df4['Cluster'])
	st.pyplot()

#####################################################################################################################################################################################
#Predictive Model
st.text(" \n")
st.header('**Predictive Model**')


html_temp = """
<div style="background-color:#1d1f96 ;padding:10px">
<h2 style="color:white;text-align:center;">Wash Item Predictive Model </h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

race = st.radio("Race", ('malay', 'chinese', 'indian', 'foreigner'))
gender = st.radio("Gender", ('male', 'female'))
body_size = st.radio("Body Size", ('moderate', 'thin', 'fat'))
age_range = st.radio("Age Range", ('20-30', '30-40', '40-50', '50-60'))
kids_category = st.radio("Kids Category" ,('no_kids', 'baby', 'toddler', 'young'))
basket_size = st.radio("Basket Size", ('small', 'big')) 
basket_colour = st.radio("Basket Colour", ('red', 'green', 'blue', 'black', 'white', 'pink', 'purple', 'yellow', 'brown', 'orange', 'grey')) 
attire = st.radio("Attire", ('casual', 'traditional', 'formal'))
shirt_colour = st.radio("Shirt Colour", ('red', 'green', 'blue', 'black', 'white', 'pink', 'purple', 'yellow', 'brown', 'orange', 'grey'))
shirt_type = st.radio("Shirt Type", ('short sleeve', 'long sleeve'))
pants_colour = st.radio("Pants Colour", ('red', 'green', 'blue', 'black', 'white', 'pink', 'purple', 'yellow', 'brown', 'orange', 'grey')) 
pants_type = st.radio("Pants Type", ('long', 'short'))
washer_no = st.radio("Washer Number", (3, 4, 5, 6))
dryer_no = st.radio("Dryer Number", (7, 8, 9, 10))
part_of_day = st.radio("Time", ('early morning', 'morning', 'afternoon', 'evening', 'night'))

X_res.drop(columns=["With_Kids"], axis=1, inplace=True)

if st.button("Predict"):
	X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3,random_state=1)
	model = svm.SVC(kernel='rbf', gamma='auto')
	model.fit(X_train, y_train)
	
	input=pd.DataFrame(np.array([[race,gender,body_size,age_range,kids_category,basket_size,basket_colour,attire,shirt_colour,shirt_type,pants_colour,pants_type,washer_no,dryer_no, part_of_day]]), columns=['Race', 'Gender', 'Body_Size', 'Age_Range', 'Kids_Category', 'Basket_Size', 'Basket_colour', 'Attire', 'Shirt_Colour', 'Shirt_Type', 'Pants_Colour', 'Pants_Type', 'Washer_No', 'Dryer_No', 'Parts_Of_The_Day'])
	input = input.astype({'Age_Range':'category', 'Washer_No': 'int64', 'Dryer_No' : 'int64'}, copy=False)
	
	input = input.apply(lambda x: dictionary[x.name].fit_transform(x))
	prediction = model.predict(input)
	prediction = dictionary['Wash_Item'].inverse_transform(prediction)

	st.write('Answer: Predicted wash item is ' + str(prediction[0]))

