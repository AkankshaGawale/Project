#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
df = pd.read_csv('DatasetPredictingPlacementinCampusRecruitment.csv')


# In[67]:


df


# In[68]:


df.shape


# In[69]:


df.describe


# In[70]:


df.info


# In[71]:


df.head()


# In[72]:


df.isnull().sum()


# In[73]:


df['salary']=df['salary'].fillna(df['salary'].median())


# In[74]:


df.isnull().sum()


# In[75]:


df.head()


# In[76]:


from sklearn.preprocessing import LabelEncoder 
cat_num =['ssc_b','gender','hsc_b','hsc_s','degree_t','workex','specialisation','status']
le = LabelEncoder()
for i in cat_num:
    df[i]= le.fit_transform(df[i])


# In[77]:


df.head()


# In[78]:


x=df.drop("status",axis=1)
y=df["status"]


# In[79]:


x.head()


# In[80]:


y.head()


# In[81]:


import seaborn as sns
sns.countplot(x="status",data=df)


# In[82]:


sns.boxplot(x="status",y="workex",data=df)


# In[83]:


sns.boxplot(x="status",y="mba_p",data=df)


# In[84]:


sns.set_theme(style="darkgrid")

ax = sns.countplot(x="degree_t", data=df)


# In[85]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=156)


# In[86]:


from sklearn.linear_model import LogisticRegression
my_model=LogisticRegression()
result=my_model.fit(x_train,y_train)


# In[87]:


predictions=result.predict(x_test)
predictions


# In[88]:


from sklearn.metrics import accuracy_score


# In[89]:


accuracy_score(y_test,predictions)


# In[90]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# In[91]:


confusion_mat=confusion_matrix(y_test,predictions)


# In[92]:


confusion_df=pd.DataFrame(confusion_mat,index=["Actual neg","Actual pos"],columns=["Predicted neg","Predicted pos"])


# In[93]:


confusion_df


# In[94]:


Color_conf_matrix=sns.heatmap(confusion_df,cmap="coolwarm",annot=True)


# In[95]:


from sklearn import metrics
print("\n**Classification Report:\n",metrics.classification_report(y_test,predictions))


# In[96]:


pred_new=my_model.predict([[0,79.00,1,55.00,0,1,98.00,2,0,66.0,1,98.78,1,760000.0]])
pred_new
                 


# In[97]:


###########Decision tree
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# In[98]:


X=df.drop("status",axis=1)
y=df["status"]
y


# In[99]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.tree import DecisionTreeClassifier
my_model = DecisionTreeClassifier(random_state=0)
result = my_model.fit(X_train,y_train)
predictions = result.predict(X_test)
predictions


# In[100]:


#CLASSIFIER SCORE
#round(roc_auc_score(y_test,predictions),5)
from sklearn.metrics import mean_absolute_error,accuracy_score


# In[101]:


mean_absolute_error(y_test, predictions)


# In[102]:


accuracy_score(y_test,predictions)


# In[103]:


from sklearn import metrics
print('\n**Classification Report:\n',metrics.classification_report(y_test,predictions))


# In[104]:


pred_new=my_model.predict([[1,79.00,1,55.00,0,1,98.00,2,0,66.0,1,98.78,1,760000.0]])
pred_new


# In[106]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[107]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[108]:


X_train


# In[109]:


type(X_train)


# In[110]:


# 3.Train the model
from sklearn.ensemble import RandomForestClassifier
my_model = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 42)
result=my_model.fit(X_train, y_train)


# In[111]:


predictions = result.predict(X_test)
X_test


# In[112]:



from sklearn import metrics


# In[113]:


print("Accuracy:",metrics.accuracy_score(y_test, predictions))


# In[116]:


from sklearn.metrics import confusion_matrix
import seaborn as sn


# In[117]:



conf_matrix =confusion_matrix(predictions,y_test)
confusion_df = pd.DataFrame(conf_matrix, index=['Actual 0','Actual 1'], columns=['Predicted 0','Predicted 1'])
sn.heatmap(confusion_df, cmap='coolwarm', annot=True)


# In[118]:


from sklearn import metrics
print('\n**Classification Report:\n',metrics.classification_report(y_test,predictions))


# In[120]:


# 5. deploy the model
pred_new = result.predict([[1,79.00,1,55.00,0,1,98.00,2,0,66.0,1,98.78,1,760000.0]])


# In[121]:


pred_new


# In[ ]:




