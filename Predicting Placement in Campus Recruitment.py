import pandas as pd
df = pd.read_csv('DatasetPredictingPlacementinCampusRecruitment.csv')

df
df.shape
df.describe
df.info
df.head()
df.isnull().sum()
df['salary']=df['salary'].fillna(df['salary'].median())
df.isnull().sum()
df.head()
from sklearn.preprocessing import LabelEncoder 
cat_num =['ssc_b','gender','hsc_b','hsc_s','degree_t','workex','specialisation','status']
le = LabelEncoder()
for i in cat_num:
    df[i]= le.fit_transform(df[i])

df.head()
x=df.drop("status",axis=1)
y=df["status"]
x.head()

y.head()
import seaborn as sns
sns.countplot(x="status",data=df)
sns.boxplot(x="status",y="workex",data=df)
sns.boxplot(x="status",y="mba_p",data=df)
sns.set_theme(style="darkgrid")

ax = sns.countplot(x="degree_t", data=df)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=156)

from sklearn.linear_model import LogisticRegression
my_model=LogisticRegression()
result=my_model.fit(x_train,y_train)
predictions=result.predict(x_test)
predictions
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
confusion_mat=confusion_matrix(y_test,predictions)

confusion_df=pd.DataFrame(confusion_mat,index=["Actual neg","Actual pos"],columns=["Predicted neg","Predicted pos"])
confusion_df
Color_conf_matrix=sns.heatmap(confusion_df,cmap="coolwarm",annot=True)

from sklearn import metrics
print("\n**Classification Report:\n",metrics.classification_report(y_test,predictions))

pred_new=my_model.predict([[0,79.00,1,55.00,0,1,98.00,2,0,66.0,1,98.78,1,760000.0]])
pred_new
                 

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import warnings


X=df.drop("status",axis=1)
y=df["status"]
y

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.tree import DecisionTreeClassifier
my_model = DecisionTreeClassifier(random_state=0)
result = my_model.fit(X_train,y_train)
predictions = result.predict(X_test)
predictions

#CLASSIFIER SCORE
#round(roc_auc_score(y_test,predictions),5)
from sklearn.metrics import mean_absolute_error,accuracy_score

mean_absolute_error(y_test, predictions)
accuracy_score(y_test,predictions)
from sklearn import metrics
print('\n**Classification Report:\n',metrics.classification_report(y_test,predictions))

pred_new=my_model.predict([[1,79.00,1,55.00,0,1,98.00,2,0,66.0,1,98.78,1,760000.0]])
pred_new

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train
type(X_train)
# 3.Train the model
from sklearn.ensemble import RandomForestClassifier
my_model = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 42)
result=my_model.fit(X_train, y_train)


predictions = result.predict(X_test)
X_test


from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, predictions))

from sklearn.metrics import confusion_matrix
import seaborn as sn

conf_matrix =confusion_matrix(predictions,y_test)
confusion_df = pd.DataFrame(conf_matrix, index=['Actual 0','Actual 1'], columns=['Predicted 0','Predicted 1'])
sn.heatmap(confusion_df, cmap='coolwarm', annot=True)

from sklearn import metrics
print('\n**Classification Report:\n',metrics.classification_report(y_test,predictions))


# 5. deploy the model
pred_new = result.predict([[1,79.00,1,55.00,0,1,98.00,2,0,66.0,1,98.78,1,760000.0]])

pred_new

