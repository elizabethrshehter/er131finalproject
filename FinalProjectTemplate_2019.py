#!/usr/bin/env python
# coding: utf-8

# # ER131 Final Project (replace this with your project title)
# Fall 2019
# 
# In this cell, give an alphabetical (by last name) list of student group members.  Beside each student's name, provide a description of each student's contribution to the project.

# ## Basic Project Requirements (delete this markdown cell in your final submission)
# 
# **How to use this notebook**:  This notebook is the template for your semester project.  Each markdown cell provides instructions on what to do in order to complete a successful project.  The cell you're reading right now is the only one you can delete from what you eventually hand in.  For the other cells:
# 1. You may replace the instructions in each cell with your own work but do not edit the cell titles (with the exception of the project title, above).  
# 2. Follow the instructions in each section carefully.  For some sections you will enter only markdown text in the existing cells. For other sections, you'll accompany the markdown cells with additional code cells, and perhaps more markdown, before moving on to the next section.  
# 
# **Grading**.  You'll see point allocations listed in each of the section titles below.  In addition, there are other categories for points: 
# 1. Visualization (10 points).  Plots should be well organized, legible, labelled, and well-suited for the question they are being used to answer or explore.  
# 2. Clarity (5 points). Note that clarity also supports points elsewhere, because if we can't understand what you're explaining, we'll assume you didn't understand what you were doing and give points accordingly!  
# 
# For each Section or Category, we will give points according to the following percentage scale:
# 1. More than 90%:  work that is free of anything but superficial mistakes, and demonstrates creativity and / or a very deep understanding of what you are doing.
# 2. 80-90%: work without fundamental errors and demonstrates a basic understanding of what you're doing.
# 3. 60-80%: work with fundamental flaws in the analysis and / or conveys that you do not understand the basics of the work you are trying to do.
# 4. Below 60%: Work that is severely lacking or incomplete.  
# 
# Note that we distinguish *mistakes* from *"my idea didn't work"*.  Sometimes you don't know if you can actually do the thing you're trying to do and as you dig in you find that you can't.  That doesn't necessarily mean you made a mistake; it might just mean you needed more information.  We'll still give high marks to ambitious projects that "fail" at their stated objective, as long as that objective was clear and you demonstrate an understanding of what you were doing and why it didn't work.
# 
# **Working in groups:**  We have the following requirements:
# 1. Projects must have at least one distinct quantitative question per student in the group.  Questions can and should be related, but they need to require distinct work efforts, and the interpretation and analysis should cover each question in detail.  If you have any doubt about whether your questions are distinct, consult with the instructors.
# 2. We use a sliding data requirements scale, see below.
# 
# **Data requirements**:  Projects must use data from a minimum of $1+N_s$ different sources, where $N_s$ is the number of students in the group.  You should merge at least two data sets. </font>
# 
# **Advice on Project Topics**:  We want you to do a project that relates to energy and environment topics.  
# 
# **Selecting a Project**: You have the choice of working on a project for a client or formulating your own project. Client project descriptions can be found in the "project" folder on Github.
# 
# **Suggested data sets**: If you choose not to work on a client projets, here are some ideas for data starting points. You can definitely bring your own data to the table!
# 1. [Purple Air](https://www.purpleair.com) Instructions on how to download PurpleAir data are [here](https://docs.google.com/document/d/15ijz94dXJ-YAZLi9iZ_RaBwrZ4KtYeCy08goGBwnbCU/edit).
# 2. California Enviroscreen database.  Available [here].(https://oehha.ca.gov/calenviroscreen/report/calenviroscreen-30) 
# 3. Several data sets available from the UC Irvine machine learning library:
#     1. [Forest Fires](https://archive.ics.uci.edu/ml/datasets/Forest+Fires)
#     4. [Climate](https://archive.ics.uci.edu/ml/datasets/Greenhouse+Gas+Observing+Network)
#     5. [Ozone](https://archive.ics.uci.edu/ml/datasets/Ozone+Level+Detection)
# 4. California Solar Initiative data (installed rooftop solar systems).  Available [here](https://www.californiasolarstatistics.ca.gov/data_downloads/).
# 5. World Bank Open Data, available [here](https://data.worldbank.org).
# 6. California ISO monitored emissions data, [here](http://www.caiso.com/TodaysOutlook/Pages/Emissions.aspx).
# 7. Energy Information Administration Residential Energy Consumption Survey, [here] (https://www.eia.gov/consumption/residential/data/2015/) 
# 
# **Dates**:
# You have the following due dates:
# 1. Thursday, October 3 (as part of homework 4): by this date you'll be expected to have formed groups of 2-3, identified your data sources, and defined a forecasting question. Feel free to use the lab time period in previous weeks to talk to potential group members, and to post on Piazza if you're looking for additional group members. The deliverable will be a question on homework 4 where you provide your group member names, data sources, and forecasting question.
# 1. Tuesday, December 17 (instead of final exam): This is when poster presentations will happen. Your deliverable is one printed poster for your group outlining the background behind your project, the questions explored, your data sources and choice of model, and results and discussion. A more detailed outline of the poster requirements will be posted later in the semester on Github.
# 1. Wednesday, December 18: Your Jupyter notebook and data should be submitted to bCourses (one submission per group). 
# 
# Throughout the semester, some of the homework assignments will include short questions designed to help you explore your project's data and analysis approach.
# 
# Ok, now on to the project!

# ## Abstract (5 points)
# Although this section comes first, you'll write it last.  It should be a ~250 word summary of your project.  1/3rd of the abstract should provide background, 1/3rd should explain what you did, and 1/3rd should explain what you learned.

# ## Project Background (5 points)
# In this section you will describe relevant background for your project.  It should give enough information that a non-expert can understand in detail the history and / or context of the system or setting you wish to study, the need for quantitative analysis, and, broadly, what impact a quantitative analyses could have on the system.  Shoot for 500 words here.

# ## Project Objective (5 points)
# In this section you will pose the central objective or objectives for your semester project.  Objectives should be extremely clear, well-defined and clearly cast as forecasting problems.  
# 
# Some example questions: 
# 1. *"The purpose of this project is to train and evaluate different models to predict soil heavy metal contamination levels across the state of Louisiana, using a variety of features drawn from EPA, the US Census, and NAICS databases."* or
# 2. *"The purpose of this project is to train and evaluate different models to predict 1-minute generation from a UCSD solar PV site, up to 2 hours into the future, using historical data as well as basic weather forecast variables.*" or
# 3. *"The purpose of this project is to forecast daily emergency room visits for cardiac problems in 4 major US cities, using a majority of features including air quality forecasts, weather forecasts and seasonal variables."*
# 
# You should reflect here on why it's important to answer these questions.  In most cases this will mean that you'll frame the answers to your questions as informing one or more *resource allocation* problems.  If you have done a good job of providing project background (in the cell above) then this reflection will be short and easy to write.
# 
# **Comment on novelty:** You may find it hard to identify a project question that has *never* been answered before.  It's ok if you take inspiration from existing analyses.  However you shouldn't exactly reproduce someone else's analysis.  If you take inspiration from another analyses, you should still use different models, different data, and so on.

# 1. Does a specific residential unit have air conditioning based on different features known about the unit?
# 
# 2. How much AC electricity consumption is a residential unit likely to use based on different features known about the unit?
# 
# 3. Out of the features in the model, are any of the demographic features more significant or less significant in predicting ownership of AC units in residential homes?
# 

# ## Input Data Description (5 points)
# Here you will provide an initial description of your data sets, including:
# 1. The origins of your data.  Where did you get the data?  How were the data collected from the original sources?
# 2. The structure, granularity, scope, temporality and faithfulness (SGSTF) of your data.  To discuss these attributes you should load the data into one or more data frames (so you'll start building code cells for the first time).  At a minimum, use some basic methods (`.head`, `.loc`, and so on) to provide support for the descriptions you provide for SGSTF. 
# 
# [Chapter 5](https://www.textbook.ds100.org/ch/05/eda_intro.html) of the DS100 textbook might be helpful for you in this section.

# ## Data Cleaning (10 points)
# In this section you will walk through the data cleaning and merging process.  Explain how you make decisions to clean and merge the data.  Explain how you convince yourself that the data don't contain problems that will limit your ability to produce a meaningful analysis from them.  
# 
# [Chapter 4](https://www.textbook.ds100.org/ch/04/cleaning_intro.html) of the DS100 textbook might be helpful to you in this section.  

# In[14]:


import math
import numpy as np
import matplotlib 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import tree


import urllib
import os.path
from shutil import copyfile


# In[2]:


# Install packages
get_ipython().system('pip install xlrd')
get_ipython().system('pip install graphviz')


# Paycool tells us whether or not they have air conditioning 
# 
# 1. 1 and 2 mean YES 
# 2. 3 means NO 
# 3. 97 means missing
# 
# We can use this as our response variable for the first classification question

# ![PAYCOOL](PAYCOOL_info.png 'PAYCOOL_info.png')

# In[3]:


#read in data and select columns
surv_data = pd.read_csv('Survdata.csv')


# In[4]:


#selecting all columns that somehow relate to AC
surv_AC = surv_data[["DWLTYPE",
"OWNRENT",
"BUILTYR",
"SEASOCC",
"SQFT",
"EXTWLINS",
"ACEILINS",
"NGUTIL",
"PAYCOOL",
"CLCNTAGE",
"avginc",
"CTEVZON",
"CLCTLTYP",
"CMRNSET",
"CDAYSET",
"CEVNSET",
"CNITESET",
"NOROOMAC",
"ACTYP1",
"ACAGE1",
"CCADD",
"CCFUEL",
"EDUC",
"ETHNIC",
"HOHIND1",
"HOHASN1",
"HOHBLK1",
"HOHLAT1",
"HOHWHT1",
"HOHOTH1",
"INCOME",
"CZT24",
"caccnt",
"RACCNT",
"rescnt",
"HOHETH",
"cecfast",
"new_cac_uec"]]

#Dropping additional columns that directly indicate AC and therefore aren't useful to predict on 
surv_AC = surv_AC.drop(columns= ['new_cac_uec','ACTYP1','CLCNTAGE','caccnt','RACCNT', 'CTEVZON',
'CLCTLTYP',
'CMRNSET',
'CDAYSET',
'CEVNSET',
'CNITESET',
'NOROOMAC',
'ACAGE1',
'CCADD',
'CCFUEL'])


# In[5]:


#issue that there are a 100 more nans than originally were missing from their data

#drop rows with 97
survey_clean = surv_AC[surv_AC['PAYCOOL'] != 97]

#number of datapoints using for response variable
num_data = sum(survey_clean['PAYCOOL'] != 97)
           
#ask about this warning
survey_clean['PAYCOOL'] = np.where(survey_clean['PAYCOOL'] == 2, 1, survey_clean['PAYCOOL'])

#check which columns contain Nans, we don't want to drop all rows containing Nans because that will reduce our dataset sample significantly
survey_clean.isna().any()

survey_clean = survey_clean.fillna(0).astype(int)

survey_clean.head()


# In[23]:


#load the column dictionary file: key for all the column names, what they mean 
coldict = pd.read_csv('coldict.csv')
coldict_clean = coldict[['Column Name', 'Notes']]
coldict_clean.columns = ['Feature', 'Definition']
coldict_clean.head()


# ### Exploring PM2.5 Concentrations By Climate Zone (CZT24) and Zipcode in Cal Enviro Screen ####

# In[22]:


ces = pd.read_csv('ces.csv') #read in the Cal Enviro Screen Data 
cztozip = pd.read_csv('climatezonesbyzip.csv') #read in Climate Zone by Zipcode from energy.ca.gov


# In[19]:


cztozip.columns = ['ZIP', 'CZT24']
czwithzip = cztozip.groupby('CZT24')['ZIP'].apply(list).to_frame()


# In[20]:


pm25byzip = ces[['ZIP', 'PM2.5']]
pm25byzip = pm25byzip.groupby(['ZIP']).mean()


# In[21]:


# add zipcode column to surv_AC by climate zone
surv_AC_withzip = surv_AC.merge(czwithzip, how = 'left', on = 'CZT24')


# ## Data Summary and Exploratory Data Analysis (10 points)
# 
# In this section you should provide a tour through some of the basic trends and patterns in your data.  This includes providing initial plots to summarize the data, such as box plots, histograms, trends over time, scatter plots relating one variable or another.  
# 
# [Chapter 6](https://www.textbook.ds100.org/ch/06/viz_intro.html) of the DS100 textbook might be helpful for providing ideas for visualizations that describe your data.  

# In[ ]:


x = survey_clean.groupby('INCOME').count().drop(97).index
height = survey_clean.groupby('INCOME').count().drop(97)['PAYCOOL']

#plt.bar(x,height,align = 'center')

sns.barplot(x, height,palette = "muted")
plt.xlabel('Income Bracket (Low to High)')
plt.ylabel('AC Units Installed ')
plt.title('Relationship between Income and AC Installation')
plt.show()


# In[ ]:


x = survey_clean.groupby('EDUC').count().drop(97).index
height = survey_clean.groupby('EDUC').count().drop(97)['PAYCOOL']
#plt.bar(x,height,align = 'center')

sns.barplot(x, height,palette = "muted")
plt.xlabel('Education level (Low to High)')
plt.ylabel('Number With AC Installed ')
plt.title('Relationship between Education and AC Installation')
plt.show()


# ## Forecasting and Prediction Modeling (25 points)
# 
# This section is where the rubber meets the road.  In it you must:
# 1. Explore at least 3 prediction modeling approaches, ranging from the simple (e.g. linear regression, KNN) to the complex (e.g. SVM, random forests, Lasso).  
# 2. Motivate all your modeling decisions.  This includes parameter choices (e.g., how many folds in k-fold cross validation, what time window you use for averaging your data) as well as model form (e.g., If you use regression trees, why?  If you include nonlinear features in a regression model, why?). 
# 1. Carefully describe your cross validation and model selection process.  You should partition your data into training, testing and *validation* sets.
# 3. Evaluate your models' performance in terms of testing and validation error.  Do you see evidence of bias?  Where do you see evidence of variance? 
# 4. Very carefully document your workflow.  We will be reading a lot of projects, so we need you to explain each basic step in your analysis.  
# 5. Seek opportunities to write functions allow you to avoid doing things over and over, and that make your code more succinct and readable. 
# 

# In[24]:


#functions for all the models we will be using

def BuildDecisionTree(X_train, X_val, y_train, y_val):
    first_tree = DecisionTreeClassifier()
    first_tree.fit(X_train, y_train)

    print("Number of features: {}".format(first_tree.tree_.n_features))
    print("Number of nodes (internal and terminal): {}".format(first_tree.tree_.node_count), "\n")

    train_score = first_tree.score(X_train, y_train)
    val_score = first_tree.score(X_val, y_val)

    print('Train Score: ', train_score)
    print('Validation Score: ', val_score)
    
    return first_tree

def TunedDecisionTree(X_train, X_val, y_train, y_val, max_depth, max_features, max_leaf_nodes):
    opt_tree = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features, max_leaf_nodes=max_leaf_nodes)
    opt_tree.fit(X_train, y_train)

    print("Number of features: {}".format(opt_tree.tree_.n_features))
    print("Number of nodes (internal and terminal): {}".format(opt_tree.tree_.node_count), "\n")

    train_score = opt_tree.score(X_train, y_train)
    val_score = opt_tree.score(X_val, y_val)

    print('Train Score: ', train_score)
    print('Validation Score: ', val_score)
    return opt_tree

def BaggingTree(X_train, X_val, y_train, y_val):

    from sklearn.ensemble import BaggingClassifier

    bag_tree = BaggingClassifier()
    bag_tree.fit(X_train, y_train)

    bag_train_score = bag_tree.score(X_train, y_train)
    bag_val_score = bag_tree.score(X_val, y_val)

    print('Train Score: ', bag_train_score)
    print('Validation Score: ', bag_val_score)
   
    return bag_tree

def RandomForest(X_train, X_val, y_train, y_val):

    from sklearn.ensemble import RandomForestClassifier

    rf_tree = RandomForestClassifier()
    rf_tree.fit(X_train, y_train)

    rf_train_score = rf_tree.score(X_train, y_train)
    rf_val_score = rf_tree.score(X_val, y_val)

    print('Train Score: ', rf_train_score)
    print('Validation Score: ', rf_val_score)
    return rf_tree

def opt_RandomForest(X_train, X_val, y_train, y_val, max_depth, max_features, max_leaf_nodes,min_samples_leaf):

    rf_opt_tree = RandomForestClassifier(max_depth = max_depth, max_features = max_features, max_leaf_nodes = max_leaf_nodes, min_samples_leaf = min_samples_leaf)
    rf_opt_tree.fit(X_train, y_train)

    rf_train_score = rf_opt_tree.score(X_train, y_train)
    rf_val_score = rf_opt_tree.score(X_val, y_val)

    print('Train Score: ', rf_train_score)
    print('Validation Score: ', rf_val_score)

    return rf_opt_tree

def ConfusionMatrix(rf_opt_tree, X_test, y_test):
    from sklearn.metrics import classification_report, confusion_matrix 
    y_pred = rf_opt_tree.predict(X_test)
    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test,y_pred)) 

def ModelTestScores(X_test, y_test, first_tree, opt_tree, bag_tree, rf_tree, rf_opt_tree):
    models = [first_tree, opt_tree, bag_tree, rf_tree, rf_opt_tree]
    for i in models:
        print('Test Score: ', i.score(X_test, y_test))
    return 


# In[25]:


#PAYCOOL is our response column 

features = survey_clean.drop(columns = ['PAYCOOL'])
target = survey_clean['PAYCOOL']

# split test set
X, X_test, y, y_test = train_test_split(features, target, random_state = 1, test_size = .2)

# split between train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 1, test_size = 0.25)


# In[26]:


#Build Decision Tree
first_tree = DecisionTreeClassifier()
first_tree.fit(X_train, y_train)

print("Number of features: {}".format(first_tree.tree_.n_features))
print("Number of nodes (internal and terminal): {}".format(first_tree.tree_.node_count), "\n")

train_score = first_tree.score(X_train, y_train)
val_score = first_tree.score(X_val, y_val)

print('Train Score: ', train_score)
print('Validation Score: ', val_score)


# To view decision tree: [Webgraphviz](http://webgraphviz.com)

# In[28]:


import graphviz
print(tree.export_graphviz(first_tree, feature_names=X.columns))


# In[30]:


#Feature Importance Table for First Tree
importance = pd.DataFrame({'Feature': X.columns, 'Importance': first_tree.feature_importances_}).sort_values(by = 'Importance',ascending = False)
importance.merge(coldict_clean,how = 'inner', on = 'Feature')


# In[31]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {'max_leaf_nodes': randint(3, 100),
              'max_features': randint(2, 22),
              'max_depth': randint(1, 10)}

rnd_search = RandomizedSearchCV(first_tree, param_distributions=param_dist, 
                                cv=10, n_iter=200, random_state = 2)
rnd_search.fit(X_train, y_train)

print(rnd_search.best_score_) 
print(rnd_search.best_params_)


# In[33]:


#New Decision Tree with tuned parameters from cross validation
opt_tree = DecisionTreeClassifier(max_depth= 9, max_features= 17, max_leaf_nodes= 58, random_state = 2)
opt_tree.fit(X_train, y_train)

print("Number of features: {}".format(opt_tree.tree_.n_features))
print("Number of nodes (internal and terminal): {}".format(opt_tree.tree_.node_count), "\n")

train_score = opt_tree.score(X_train, y_train)
val_score = opt_tree.score(X_val, y_val)

print('Train Score: ', train_score)
print('Validation Score: ', val_score)


# In[34]:


#View tree
import graphviz
print(tree.export_graphviz(opt_tree, feature_names=X.columns))


# In[35]:


importance = pd.DataFrame({'Feature': X.columns, 'Importance': opt_tree.feature_importances_}).sort_values(by = 'Importance',ascending = False)
importance.merge(coldict_clean,how = 'inner', on = 'Feature')


# In[36]:


# Bagging
from sklearn.ensemble import BaggingClassifier

bag_tree = BaggingClassifier()
bag_tree.fit(X_train, y_train)

bag_train_score = bag_tree.score(X_train, y_train)
bag_val_score = bag_tree.score(X_val, y_val)

print('Train Score: ', bag_train_score)
print('Validation Score: ', bag_val_score)


# In[ ]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_tree = RandomForestClassifier()
rf_tree.fit(X_train, y_train)

rf_train_score = rf_tree.score(X_train, y_train)
rf_val_score = rf_tree.score(X_val, y_val)

print('Train Score: ', rf_train_score)
print('Validation Score: ', rf_val_score)


# In[ ]:


#tuning hyperparameters
param_dist = {#'learning_rate': randint(1, 5),
              'max_leaf_nodes': randint(3, 100),
              'max_features': randint(2, 22),
              'max_depth': randint(1, 10),
              'min_samples_leaf': randint(1, 30)}

rnd_rf_search = RandomizedSearchCV(rf_tree, param_distributions=param_dist, 
                                cv=10, n_iter=50, random_state=1)

rnd_rf_search.fit(X_train, y_train)

print(rnd_rf_search.best_params_)


# In[ ]:


#tuned random forest

rf_opt_tree = RandomForestClassifier(max_depth = 9, max_features = 11, max_leaf_nodes = 70, min_samples_leaf = 8)
rf_opt_tree.fit(X_train, y_train)

rf_train_score = rf_opt_tree.score(X_train, y_train)
rf_val_score = rf_opt_tree.score(X_val, y_val)

print('Train Score: ', rf_train_score)
print('Validation Score: ', rf_val_score)

y_pred = rf_opt_tree.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred)) 


# In[ ]:


#Random Forest Feature Importance
feature_importance = rf_opt_tree.feature_importances_
feature_importance = feature_importance / feature_importance.max()
rf_feat = pd.DataFrame({'Feature':X_train.columns, 'Importance':feature_importance}).sort_values(by = 'Importance',ascending = False)
rf_feat.head()


# In[ ]:


plt.figure(figsize=(7, 7))
plt.barh(width=rf_feat.Importance, y=rf_feat.Feature);
plt.title('Feature Importance (Random Forest)')
plt.show()


# In[ ]:


#SVM Model
from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train, y_train.values.ravel())

y_pred = svc.predict(X_test)
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))


# In[ ]:


#KNN Nearest Neighbors Classifier 
#Hypertuning KNN Nearest Neighbor Classifier 

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier 
#create new a knn model
knn = KNeighborsClassifier()
#create dict of all values to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 20)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X_train, y_train)


# In[ ]:


#Use Optimal Number of Neighbors

knnclassifier = KNeighborsClassifier(n_neighbors=knn_gscv.best_params_['n_neighbors'])

# Train the model using the training sets
knnclassifier.fit(X_train,y_train)

#Predict Output
y_pred= knnclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))


# In[ ]:


models = [first_tree, opt_tree, bag_tree, rf_tree, rf_opt_tree, svc, knnclassifier]
for i in models:
    print('Test Score: ', i.score(X_test, y_test))


# ## Part 2: PM2.5 Prediction Analysis 

# In[44]:


# Head of Household Ethnicity 
#0 = Other, 1 = Native American, 2 = Asian/Pacific-Islander , 3 = African-American, 4 = Hispanic, 5 = Caucasian

survey_clean.loc[survey_clean['HOHOTH1']==1,'HOHOTH1']=0
survey_clean.loc[survey_clean['HOHASN1']==1,'HOHASN1']=2
survey_clean.loc[survey_clean['HOHBLK1']==1,'HOHBLK1']=3
survey_clean.loc[survey_clean['HOHLAT1']==1,'HOHLAT1']=4
survey_clean.loc[survey_clean['HOHWHT1']==1,'HOHWHT1']=5


ETH = survey_clean[['HOHOTH1','HOHIND1','HOHASN1','HOHBLK1','HOHLAT1','HOHWHT1']]
ETH['HOHETH1'] = ETH.sum(axis=1)
ETH.loc[ETH['HOHETH1']>5, 'HOHETH1']=0
#ETH['HOHETH1'].unique()


# In[49]:


#Education 
#In Cal Enviro Screen Data: Percent of population over 25 with less than a high school education (in Cal Enviro Screen)

#In RASS Data: 1 = Elementary, 2 = Some high school, 3=High school graduate, 4 = Some college/trade/vocational school, 5 = College graduate, 6 = Postgraduate degree, 97 = Nan  

# We are going to build a model that classifies based on just Education
# The X training data will consist of one feature, 1 = Education with less than a highschool degree, 2 = Education with more than a highschool degree (Assume those who filled out the survey are 25 or older)

data = survey_clean[['EDUC', 'PAYCOOL']]

data = data[data['EDUC'] != 97]

data['EDUC'] = np.where((data['EDUC'] == 1) | (data['EDUC'] == 2), 1, data['EDUC'])
data['EDUC'] = np.where((data['EDUC'] == 3) | (data['EDUC'] == 4)| (data['EDUC'] == 5)| (data['EDUC'] == 6), 2, data['EDUC'])


# In[69]:


#Add Head of Household Ethnicity into Data 

data['HOHETH'] = ETH['HOHETH1']

#Split the data

# features are Education and Head of Household
features = data[['EDUC', 'HOHETH']]
target = data['PAYCOOL']
# split test set
X2, X2_test, y2, y2_test = train_test_split(features, target, random_state = 1, test_size = .2)

# split between train and validation sets
X2_train, X2_val, y2_train, y2_val = train_test_split(X2, y2, random_state = 1, test_size = 0.25)


# In[75]:


#Visualize the data

#plt.figure(figsize = (10,7))

#colormap = np.array(['red', 'blue'])
#classes = pd.unique(target)


#plt.title('AC Adoption vs. Head of Household Ethnicity and Education')
#plt.xlabel('AC Adoption (1 = Yes, 3 = No)')
#plt.ylabel('Classification Values for HOHETH & EDUC')
#plt.legend()
#plt.show()


# In[67]:


#Simple first decision tree
BuildDecisionTree(X2_train, X2_val, y2_train, y2_val)


# In[ ]:





# ## Interpretation and Conclusions (20 points)
# In this section you must relate your modeling and forecasting results to your original research question.  You must 
# 1. What do the answers mean? What advice would you give a decision maker on the basis of your results?  How might they allocate their resources differently with the results of your model?  Why should the reader care about your results?
# 2. Discuss caveats and / or reasons your results might be flawed.  No model is perfect, and understanding a model's imperfections is extremely important for the purpose of knowing how to interpret your results.  Often, we know the model output is wrong but we can assign a direction for its bias.  This helps to understand whether or not your answers are conservative.  
# 
# Shoot for 500-1000 words for this section.

# In[ ]:




