#!/usr/bin/env python
# coding: utf-8

# # 1. Importing Liabraries

# Importing Dependencies

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# Importing Models

# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# # 2. Uploading the dataset

# In[3]:


# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv') 


# # 3. Exploratory Data Analysis

# 3.1.1) Head of the dataset

# In[4]:


# printing the first 10 rows of the dataset
diabetes_dataset.head(10)


# In[5]:


#dispay last 10records of the dataset
diabetes_dataset.tail(10)


# In[6]:


#display randomly any number of records of data
diabetes_dataset.sample(5)


# 3.1.2) Shape of the dataset

# In[7]:


# number of rows and Columns in this dataset
diabetes_dataset.shape


# no. of rows = 768
# 
# no. of columns = 9

# 3.1.3) List of types of columns
# 
# Using diabetes_dataset.dtypes, we get the list of all the columns in our dataset.

# In[8]:


#list of the types of dataset
diabetes_dataset.dtypes


# 3.1.4) Information of the dataset
# 
# info() is used to check the information about the data and the datatypes of each repective attribute

# In[9]:


#finding out if the dataset contains any null value
diabetes_dataset.info()


# 3.1.5) Summary of the dataset
# 
# we can clearly see the minimum value, mean, max values, etc

# In[10]:


# getting the statistical measures of the data
diabetes_dataset.describe()


# Observation:
# 
# In the above table, min value of some columns is 0.
# 
# 

# 3.2) Data Cleaning
# 
# 3.2.1) Drop the duplicates
# 
# firstly checking if any dataset exist, and if it does then it should be removed from dataframe

# In[11]:


#checkthe shape before removing duplicates
diabetes_dataset.shape


# In[12]:


diabetes_dataset = diabetes_dataset.drop_duplicates()


# In[13]:


#check the shape after removing duplicates
diabetes_dataset.shape


# No duplicates in the dataset

# 3.2.2) Check the NULL values
# 
# using isnull.sum() function we can see the null values present in the every column in dataset.

# In[14]:


# Identify missing values
missing_values = diabetes_dataset.isnull().sum()

# Print the count of missing values for each column
print("Missing Values:\n", missing_values)


# There is no null value in the dataset

# In[15]:


diabetes_dataset.columns


# Check the number of zero values in the dataset

# In[16]:


print('No. of zero values in pregnancies ', diabetes_dataset[diabetes_dataset['Pregnancies'] == 0].shape[0])


# In[17]:


print('No. of zero values in glucose ', diabetes_dataset[diabetes_dataset['Glucose'] == 0].shape[0])


# In[18]:


print('No. of zero values in blood pressure ', diabetes_dataset[diabetes_dataset['BloodPressure'] == 0].shape[0])


# In[19]:


print('No. of Glucose values in skin Thickness ', diabetes_dataset[diabetes_dataset['SkinThickness'] == 0].shape[0])


# In[20]:


print('No. of zero values in insulin ', diabetes_dataset[diabetes_dataset['Insulin'] == 0].shape[0])


# In[21]:


print('No. of zero values in BMI ', diabetes_dataset[diabetes_dataset['BMI'] == 0].shape[0])


# Replace no. of zeros values with mean of that columns

# In[22]:


diabetes_dataset['Glucose']=diabetes_dataset['Glucose'].replace(0, diabetes_dataset['Glucose'].mean())
print('No. of zero values in glucose ', diabetes_dataset[diabetes_dataset['Glucose'] == 0].shape[0])


# In[23]:


diabetes_dataset['BloodPressure']=diabetes_dataset['BloodPressure'].replace(0, diabetes_dataset['BloodPressure'].mean())
diabetes_dataset['SkinThickness']=diabetes_dataset['SkinThickness'].replace(0, diabetes_dataset['SkinThickness'].mean())
diabetes_dataset['Insulin']=diabetes_dataset['Insulin'].replace(0, diabetes_dataset['Insulin'].mean())
diabetes_dataset['BMI']=diabetes_dataset['BMI'].replace(0, diabetes_dataset['BMI'].mean())


# In[24]:


diabetes_dataset.describe()


# # 4. Data Visualization

# 4.1) Count Plot

# In[25]:


# outcome count plot
f, ax = plt.subplots(1,2,figsize=(10, 5))
diabetes_dataset['Outcome'].value_counts().plot.pie(explode=[0,0.1],autopct = '%1.1f%%', ax=ax[0],shadow=True)
ax[0].set_title('Outcome')
ax[0].set_ylabel('')
sns.countplot('Outcome',data=diabetes_dataset, ax=ax[1])
ax[1].set_title('Outcome')
N,P = diabetes_dataset['Outcome'].value_counts()
print('Negative (0): ', N)
print('Positive (1): ', P)
plt.grid()
plt.show()


# Out of total 768 people, 268 are diabetic and 500 are non-diabetic In the outcomecolumn, 1--> DIABETES POSITIVE and 0--> DIABETES NEGATIVE. The cout Plot tells us that the dataset is imbalanced, as number of patients who don't have diabetes is more than those who don't have diabetes.

# 4.2) Histograms
# 
# Histogram are one of the most common graphs used to display numeric data. Distribution of the data - whether the data is normally distributed or it is skewed (to the left or right)

# In[26]:


#histogram of each feature
diabetes_dataset.hist(bins=10, figsize=(10,10))
plt.show()


# 4.3) Pairplot This will create scatterplots betweeen all of our varibles.

# In[27]:


sns.pairplot(diabetes_dataset, hue='Outcome')
plt.suptitle('Pairplot of Diabetes Dataset', y=1.02, fontsize=16)
plt.show()


# 4.5) Analyzing relationships between varibles

# Correlation analysis is used to quantify the degree to which two varibale are related. through this, you evaluate correlatiom coefficient thattells you how much one varibale changes when the other one does.

# In[28]:


correlation_matrix = diabetes_dataset.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# # 5. Split the dataframe into x and y

# In[29]:


target_name = 'Outcome'
#seperate object for target feature
y = diabetes_dataset[target_name]

#seperate Object for Input features
x = diabetes_dataset.drop(target_name, axis=1)


# In[30]:


x.head()


# In[31]:


y.head()


# # 6. Removing the outliers

# An outlier is an extremely high or extremely low data point relative to the nearest data point and the rest of the neighboring co-existing values in a data graph or dataset you're working with. These extreme values can impact your statistical power as well, making it hard to detect a true effect if there is one.

# In[32]:


# Calculate Z-scores for each numerical column
z_scores = zscore(diabetes_dataset)

# Define a threshold for Z-scores (e.g., 3 standard deviations)
threshold = 3

# Identify rows with outliers
outlier_rows = (abs(z_scores) > threshold).any(axis=1)

# Remove rows with outliers
diabetes_dataset_no_outliers = diabetes_dataset[~outlier_rows]

# Display the modified dataset without outliers
print("Original dataset shape:", diabetes_dataset.shape)
print("Dataset shape after removing outliers:", diabetes_dataset_no_outliers.shape)


# # 7. Cross Validation

# In[33]:


#list of models
models = [LogisticRegression(max_iter = 1000), SVC(kernel = 'linear'), KNeighborsClassifier(), RandomForestClassifier()]


# In[34]:


def compare_models_cross_validation():
    for model in models:
        
        cv_score = cross_val_score(model, x, y, cv=5)
        
        mean_accuracy = sum(cv_score)/len(cv_score)
        
        mean_accuracy = mean_accuracy*100
        
        mean_accuracy = round(mean_accuracy, 2)
        
        print('Cross Validation accuracies for ', model, '= ', cv_score)
        print('Accuracy % of the ', model, mean_accuracy)
        print('---------------------------------------------------------------------------')


# In[35]:


compare_models_cross_validation()


# Logistic Regression classifier has the highest accuracy score

# # 8. Confusion Matrix

# In[36]:


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

def compare_models_cross_validation_with_confusion_matrix():
    for model in models:
        # Perform cross-validation and get predictions
        y_pred = cross_val_predict(model, x, y, cv=5)
        
        # Compute confusion matrix
        cm = confusion_matrix(y, y_pred)
        print("Confusion matrix for", model, ":\n", cm)
        
        # Compute accuracy
        accuracy = np.trace(cm) / float(np.sum(cm))
        print('Accuracy of', model, ':', accuracy)
        print('---------------------------------------------------------------------------')

compare_models_cross_validation_with_confusion_matrix()


# In[37]:


import seaborn as sns

def compare_models_cross_validation_with_confusion_matrix():
    for model in models:
        # Perform cross-validation and get predictions
        y_pred = cross_val_predict(model, x, y, cv=5)
        
        # Compute confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.title('Confusion Matrix for ' + str(model))
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()
        
        # Compute accuracy
        accuracy = np.trace(cm) / float(np.sum(cm))
        print('Accuracy of', model, ':', accuracy)
        print('---------------------------------------------------------------------------')

compare_models_cross_validation_with_confusion_matrix()


# we can now calculate Precision, Recall and F1 score
# 
# Precision =  True Poitive / (True Positive + False Positive)
# 
# Recall = True Poitive / (True Positive + False negative)
# 
# F1 score = 2 * (precision * recall) / (precision + recall)
# 
# 

# In[38]:


from sklearn.metrics import precision_score, recall_score, f1_score

def compare_models_cross_validation_with_confusion_matrix():
    for model in models:
        # Perform cross-validation and get predictions
        y_pred = cross_val_predict(model, x, y, cv=5)
        
        # Compute confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.title('Confusion Matrix for ' + str(model))
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()
        
        # Compute accuracy
        accuracy = np.trace(cm) / float(np.sum(cm))
        print('Accuracy of', model, ':', accuracy)
        
        # Compute precision, recall, and F1-score
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        
        print('Precision of', model, ':', precision)
        print('Recall of', model, ':', recall)
        print('F1-score of', model, ':', f1)
        print('---------------------------------------------------------------------------')

compare_models_cross_validation_with_confusion_matrix()


# In[39]:


X = np.asarray(x)
Y = np.asarray(y)


#  # 9. Hyperparmeter Optimization & Model Selection

# In[40]:


#list of models
models_list = [LogisticRegression(max_iter = 10000), SVC(), KNeighborsClassifier(), RandomForestClassifier()]


# In[41]:


#creating a dictionary that contains hyperparameter values for the above mentioned models

model_hyperparameters = {
    
    'log_reg_hyperparameters': {
        
        'C': [1, 5, 10, 20]
    },
    
    
    'svc_hyperparameters': {
        
        'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
        'C' : [1, 5, 10, 20]
    },
    
    'KNN_hyperparameters' : {
        
        'n_neighbors' : [3, 5, 10]
    },
    
    
    'random_forest_hyperparameters' : {
        
        'n_estimators' : [10, 20, 50, 100]
    }
    
}


# In[42]:


type(model_hyperparameters)


# In[43]:


print(model_hyperparameters.keys())


# In[44]:


model_hyperparameters['svc_hyperparameters']


# In[45]:


model_keys = list(model_hyperparameters.keys())
print(model_keys)


# In[46]:


model_keys[1]


# In[47]:


model_hyperparameters[model_keys[1]]


# Applying GridSearchCV

# In[48]:


def ModelSelection(list_of_models, hyperparameters_dictionary):

  result = []

  i = 0

  for model in list_of_models:
    
    key = model_keys[i]
    
    params = hyperparameters_dictionary[key]
    
    i += 1
    
    print(model)
    print(params)
    print('-------------------------------------')
    
    classifier = GridSearchCV(model, params, cv = 5)
    
    #fitting the data to classifier
    classifier.fit(X, Y)
    
    
    result.append({
        'model used' : model,
        'hieghst score' : classifier.best_score_,
        'best hyperparameter' : classifier.best_params_
    })
    
  result_dataframe = pd.DataFrame(result, columns =['model used', 'hieghst score', 'best hyperparameter'])
    
  return result_dataframe
    


# In[49]:


ModelSelection(models_list, model_hyperparameters)


# SVC() with 'C': 10, 'kernel': 'linear'

# # 10. Sample Predictions

# In[50]:


# Define the input data
input_data = [(3, 126, 88, 41, 235, 39.3, 0.704, 27)]

# Iterate through each model
for model in models_list:
    # Train the model on the entire dataset
    model.fit(X, Y)
    
    # Make predictions on the input data
    prediction = model.predict(input_data)
    
    # Print the model's prediction
    print("Prediction using", type(model).__name__, ":", prediction)


# In[51]:


# Define the new input data
input_data = [(1, 182, 69, 24, 845, 30.1, 0.398, 59)]

# Iterate through each model
for model in models_list:
    # Train the model on the entire dataset
    model.fit(X, Y)
    
    # Make predictions on the input data
    prediction = model.predict(input_data)
    
    # Print the model's prediction
    print("Prediction using", type(model).__name__, ":", prediction)


# In[52]:


from sklearn.preprocessing import StandardScaler

# Define the input data
input_data = [(3, 126, 88, 41, 235, 39.3, 0.704, 27)]

# Standardize the input data using the same scaler used during preprocessing
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Load the trained Random Forest Classifier model
best_model = RandomForestClassifier(n_estimators=100)

# Fit the model with the entire dataset (assuming the model was trained on the entire dataset)
best_model.fit(X, Y)

# Make predictions on the standardized input data
predictions = best_model.predict(input_data_scaled)

# Output the prediction
if predictions[0] == 0:
    print("The model predicts that the patient does not have diabetes.")
else:
    print("The model predicts that the patient has diabetes.")


# # 11. Trends and correlations between attributes

# In[53]:


plt.figure(figsize=(10, 6))
sns.histplot(data=diabetes_dataset, x='Age', hue='Outcome', multiple='stack', bins=30, palette='muted')
plt.title('Histogram of Age for Diabetic and Non-Diabetic Individuals')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# Age vs diabetes: age is a major factor of diabetes as in the early age of patient’s chances 
# of not having diabetes are very high but as the age increases this ramp of the histogram 
# also decreases in the case of non-diabetic patients. This concludes people at a young 
# age are very rarely prone to diabetes.

# In[54]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Glucose', y='Insulin', hue='Outcome', data=diabetes_dataset, palette='viridis')
plt.title('Scatter Plot of Glucose vs Insulin colored by Outcome')
plt.xlabel('Glucose')
plt.ylabel('Insulin')
plt.show()


# Glucose vs Insulin: most of the patients positive with diabetes have high glucose or
# insulin whereas patients with less or normal glucose and insulin are non-diabetic.
# 

# In[55]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='BMI', hue='Outcome', data=diabetes_dataset, palette='viridis')
plt.title('Scatter Plot of Age vs BMI colored by Outcome')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.show()


# Age vs BMI: older people who are diabetic have higher BMI level as compared to non-diabetic patient

# In[56]:


# Box plot to show the distribution
plt.figure(figsize=(12, 8))
sns.boxplot(x='Outcome', y='Pregnancies', data=diabetes_dataset)
plt.title('Distribution of Pregnancies by Diabetes Outcome')
plt.show()


# Females with a large number of pregnancies are predicted with diabetes positively. This 
# could be visualized using a box plot representation.

# In[57]:


# Box plot to show the distribution
plt.figure(figsize=(12, 8))
sns.boxplot(x='Outcome', y='DiabetesPedigreeFunction', data=diabetes_dataset)
plt.title('Distribution of Diabetes Pedigree Function by Diabetes Outcome')
plt.xlabel('Outcome (0: Non-Diabetic, 1: Diabetic)')
plt.ylabel('Diabetes Pedigree Function')
plt.show()


# It is also noted that if diabetes pedigree function is greater than 0.5 then the chances 
# of having diabetes is more. This could also be shown in box plot.

# In[58]:


pip install xgboost


# In[59]:


from xgboost import XGBClassifier, XGBRegressor


# Adding XGBoost to the Models List.
# Integrate XGBoost into your models list.

# In[60]:


# Updated list of models
models = [LogisticRegression(max_iter=1000), SVC(kernel='linear'), KNeighborsClassifier(), RandomForestClassifier(), XGBClassifier()]

def compare_models_cross_validation():
    for model in models:
        cv_score = cross_val_score(model, x, y, cv=5)
        mean_accuracy = sum(cv_score) / len(cv_score)
        mean_accuracy = mean_accuracy * 100
        mean_accuracy = round(mean_accuracy, 2)
        print('Cross Validation accuracies for ', model, '= ', cv_score)
        print('Accuracy % of the ', model, mean_accuracy)
        print('---------------------------------------------------------------------------')

compare_models_cross_validation()


# Include hyperparameters for XGBoost and update your model selection function.

# In[64]:


# Adding hyperparameters for XGBoost
model_hyperparameters = {
    'log_reg_hyperparameters': {'C': [1, 5, 10, 20]},
    'svc_hyperparameters': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [1, 5, 10, 20]},
    'KNN_hyperparameters': {'n_neighbors': [3, 5, 10]},
    'random_forest_hyperparameters': {'n_estimators': [10, 20, 50, 100]},
    'xgboost_hyperparameters': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}

# Updated list of models
models_list = [LogisticRegression(max_iter=10000), SVC(), KNeighborsClassifier(), RandomForestClassifier(), XGBClassifier()]

# Applying GridSearchCV
def ModelSelection(list_of_models, hyperparameters_dictionary):
    result = []
    model_keys = list(hyperparameters_dictionary.keys())
    i = 0

    for model in list_of_models:
        key = model_keys[i]
        params = hyperparameters_dictionary[key]
        i += 1
        print(model)
        print(params)
        print('-------------------------------------')

        classifier = GridSearchCV(model, params, cv=5)
        classifier.fit(X, Y)

        result.append({
            'model used': model,
            'highest score': classifier.best_score_,
            'best hyperparameter': classifier.best_params_
        })

    result_dataframe = pd.DataFrame(result, columns=['model used', 'highest score', 'best hyperparameter'])
    return result_dataframe

ModelSelection(models_list, model_hyperparameters)


# Model Evaluation with XGBoost.
# Update model evaluation function to include XGBoost.

# In[65]:


def compare_models_cross_validation_with_confusion_matrix():
    for model in models:
        y_pred = cross_val_predict(model, x, y, cv=5)
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.title('Confusion Matrix for ' + str(model))
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()
        accuracy = np.trace(cm) / float(np.sum(cm))
        print('Accuracy of', model, ':', accuracy)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        print('Precision of', model, ':', precision)
        print('Recall of', model, ':', recall)
        print('F1-score of', model, ':', f1)
        print('---------------------------------------------------------------------------')

compare_models_cross_validation_with_confusion_matrix()


# Final Predictions with XGBoost.
# Make predictions using the best model, including XGBoost.

# In[66]:


# Define the input data
input_data = [(3, 126, 88, 41, 235, 39.3, 0.704, 27)]

# Iterate through each model
for model in models_list:
    model.fit(X, Y)
    prediction = model.predict(input_data)
    print("Prediction using", type(model).__name__, ":", prediction)


# Visualization of Model Performance

# In[ ]:




