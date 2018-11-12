#""" Library Imports and Dependency Management """  

import pandas as pd
import numpy as np
from sklearn import tree, model_selection
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


#""" Loading the Data Set (as a CSV file) """

file = r'C:\Users\msastry\Desktop\Data Sets\Fare Prediction\Data\Train_Modified.csv'
dataframe = pd.read_csv(file)
#dataframe.head()

#""" Discarding irrelevant features """

dataframe.drop(['id','date_book', 'date_dept', 'time_dept', 'Weekend Booking'],1,inplace = True)
#dataframe.head()

#""" Identifying Correlations between Features and Labels """ 
correlation_matrix = dataframe.apply(lambda x:pd.factorize(x)[0]).corr()
print()
print("Correlation Matrix - Values")
print(correlation_matrix)
print()
fig = plt.figure(figsize = (16,8))
axis = fig.add_subplot(111)
graph = axis.matshow(correlation_matrix)
plt.title('Correlation Matrix - Visualization')
fig.colorbar(graph)
plt.xlabel('Feature Set')
plt.ylabel('Feature Set')
plt.show()


#""" Conversion of Categorical Data into Numeric Data """
    
new_df = pd.get_dummies(dataframe, columns = ['origin', 'destination','Weekend Travel', 'Peak Hour','TimeOfDay','InTimeBooking','Day of Travel'])
#new_df.head()
    
#""" Creation of Feature Set and Labels (for classification)"""
    
X = np.array(new_df.drop(['fare_choice'],1))
y = np.array(new_df['fare_choice'])
#print(X)
#print(y)
    
#""" Train - Test Split : 70 - 30 """      
    
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size = 0.3)

#""" Feature Scaling and Normalization """      
    
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#""" Model Definition and Fitting """
    
#Pure Decision Tree with a P()=1 at each of the leaf nodes
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
#print('Training Successful')

#""" Model Verfification """

#Classification Accuracy
    
predicted_values = clf.predict(X_test)
accuracy = accuracy_score(y_test, predicted_values) # or accuracy = clf.score(X_test,y_test) 
print()
print('Predicted Values:')
print(predicted_values)
print()
print('Model Accuracy:')
print(accuracy)
    
#Comparing Predicted and Actual Values
   
df_predicted = pd.DataFrame(predicted_values)
df_actual = pd.DataFrame(y_test)
compared_values = pd.concat([df_actual, df_predicted], keys=['Actual Fare Choices', 'Predicted Fair Choices'], axis = 1)
print()
print('Actual vs. Predicted Fare Choices')
print(compared_values)
#Write the values to a file
compared_values.to_csv("ResultFile.csv")

#Confusion Matrix    

cm = confusion_matrix(y_test, predicted_values)
print()
print('Confusion Matrix - Values :')
print(cm)
print()
fig = plt.figure(figsize = (16,8))
axis = fig.add_subplot(111)
graph = axis.matshow(cm)
plt.title('Confusion Matrix of the Classifier - Visualization')
fig.colorbar(graph)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Calculation of Log Loss - another metric that quantifies the performance of a calssifier - is left for the reader to explore 
#Classification Accuracy and Log Loss functions have an interesting relationship (they can be optimized simultaneously only by tweaking the parameters of the classifier)

# """ A note on Recursive Feature Elimination - a standard approach for feature rich classification """
 
#We encourage the users to try the RFE techniques and compare the results with the standard classifiers 

# from sklearn.model_selection import StratifiedKFold
# from sklearn.feature_selection import RFECV 
# rfecv =  RFECV(estimator=clf, step=1, cv=StratifiedKFold(2),scoring='accuracy')
# X = np.array(new_df.drop(['fare_choice'],1))
# y = np.array(new_df['fare_choice'])
# rfecv.fit(X,y)
# print(rfecv.n_features_)
# plt.figure(figsize = (10,5))
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.xlabel('No of Features')
# plt.ylabel('Corss Validation Scores (Number of Correct Classifications)')
# plt.show()
# rfecv.score(X_test,y_test) 