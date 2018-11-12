#-----------------------Beginning of BLOCK ONE - Run as one Set --------------------------------------------------------------------

#""" Library Imports and Dependency Management """ 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import seaborn as sns 
color = sns.color_palette()
sns.set(rc={'figure.figsize':(20,10)})

from plotly.offline import plot
import plotly.graph_objs as go

from sklearn import model_selection
from sklearn.preprocessing import StandardScaler  
import xgboost as xgb
import math

import warnings
warnings.filterwarnings("ignore")


#""" Loading the Data Set (as a CSV file) """
file = r'C:\Users\msastry\Desktop\Data Sets\Play Store Analytics\Data\googleplaystore.csv'
df = pd.read_csv(file)
df.drop_duplicates(subset='App', inplace=True)
df.head()
print('Number of Applications in the Data Set:', len(df))

#"""Data Cleaning"""

#Eliminating stray entries in the 'Installs' and 'Android Ver' columns 
df = df[df['Android Ver'] != 'NaN']
df = df[df['Installs'] != 'Free']

 

#'Installs' Column
#Removing ',' and '+' signs
df['Installs'] = df['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: float(x))

#'Size' Column
#Removing 'M' and 'K' + changing the scale 
df['Size'] = df['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: float(x))

#'Price' Column
#Removing '$' signs
df['Price'] = df['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
df['Price'] = df['Price'].apply(lambda x: float(x))

#'Reviews' Column
#Changing data type from object to an int
df['Reviews'] = df['Reviews'].apply(lambda x: int(x))

df.head()


#Overview of the data - plotting number of applications and average rating by Category 
categories = df['Category'].value_counts()
pie_chart = [go.Pie(
        labels = categories.index,
        values = categories.values,
        hoverinfo = 'value + label')]
print()
print('Total number of categories:', categories.count())
print('Count by categories:')
print(categories)
print()
print('App count by Category - Pie Chart')
plot(pie_chart, filename = "PieChart.html")


print('Average App Rating = ', np.nanmean(df['Rating']))
design = {'title' : 'Categories vs. Average Ratings',
        'xaxis': {'tickangle':-45},
        'yaxis': {'title': 'Rating'}
          }
data = [{
    'y': df.loc[df.Category==category]['Rating'], 
    'type':'box',
    'name' : category,
    'showlegend' : False
    } for i,category in enumerate(list(set(df['Category'])))]
print('Average Rating vs. Category')
plot(data, design, filename = "BoxPlot.html")

#-----------------------End of BLOCK ONE - Run as one Set --------------------------------------------------------------------



#-----------------------Beginning of BLOCK TWO - Run as one Set --------------------------------------------------------------------

#Basic EDA - Pair Plots 
rating = df['Rating'].dropna()
size = df['Size'].dropna()
installs = df['Installs'][df.Installs!=0].dropna()
reviews = df['Reviews'][df.Reviews!=0].dropna()
type = df['Type'].dropna()
print()
print('Pari-wise plot of key numeric features:')
pairplot = sns.pairplot(pd.DataFrame(list(zip(rating, np.log10(installs), size, np.log10(reviews), type)),
                                     columns=['Rating', 'Installs', 'Size','Reviews', 'Type']), hue='Type', palette="Set1")

#-----------------------End of BLOCK TWO - Run as one Set --------------------------------------------------------------------



#-----------------------Beginning of BLOCK THREE - Run as one Set --------------------------------------------------------------------

#Join Plots - Numeric Plots
sns.set_style("ticks")

plt.figure(figsize = (10,10))
print('Join Plot of Rating vs. Size')
size_vs_rating = sns.jointplot(df['Size'], df['Rating'], kind = 'kde', color = "orange", size = 8)
print('Join Plot of Rating vs. Review')
review_vs_rating = sns.jointplot(df['Reviews'], df['Rating'], kind = 'reg', color = "orange", size = 8)
print('Join Plot of Rating vs. Installs')
installs_vs_rating = sns.jointplot(df['Installs'], df['Rating'], kind ='reg', color = "orange", size = 8)

#-----------------------End of BLOCK THREE - Run as one Set --------------------------------------------------------------------



#-----------------------Beginning of BLOCK FOUR - Run as one Set --------------------------------------------------------------------

#"""Prediction of Rating Range - a Regression Problem"""

#Ignore features irrelevant for prediction  
df.drop(['App', 'Last Updated', 'Current Ver'], 1, inplace = True)
df.head()

#Visualizing Correlation 
fig = plt.figure(figsize = (16,8))
#corr = plt.matshow(df.apply(lambda x:pd.factorize(x)[0]).corr(), fignum = 1)
print()
print('Correlation Graph')
corr = df.corr(method = 'spearman', min_periods = 5)
plot = plt.matshow(corr, fignum = 1)
fig.colorbar(plot)

#-----------------------End of BLOCK FOUR - Run as one Set --------------------------------------------------------------------



#-----------------------Beginning of BLOCK FIVE - Run as one Set --------------------------------------------------------------------

new_df = pd.get_dummies(df, columns = ['Category', 'Content Rating', 'Genres', 'Android Ver', 'Type'])
new_df.replace('?', -9999, inplace = True )
new_df.dropna(inplace = True)
new_df.head()

X = np.array(new_df.drop(['Rating'],1))
y = np.array(new_df['Rating'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size = 0.3)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#-----------------------End of BLOCK FIVE - Run as one Set --------------------------------------------------------------------



#-----------------------Beginning of BLOCK SIX - Run as one Set --------------------------------------------------------------------

model = xgb.XGBRegressor(n_estimators=70, learning_rate=0.08, gamma=0, subsample=0.70,
                           colsample_bytree=1, max_depth=6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) 
print('Results - Stage 1:')
print(result.head()) 

diff_list = abs(result['Predicted']-result['Actual'])
mean_diff = np.nanmean(diff_list)
variance = np.square(diff_list-mean_diff).sum()/len(diff_list)
standard_deviation = np.sqrt(variance)
lower_bound = np.around(y_pred - (standard_deviation/2),2)
upper_bound = np.around(y_pred + (standard_deviation/2),2)

result_final = pd.DataFrame({'Actual Rating':y_test, 'Lower Bound on Predicted Rating': lower_bound, 'Upper Bound on Predicted Rating': upper_bound })
print('Final Result Set:')
print(result_final)
print('Look into the file for more clarity on the results!')
result_final.to_csv('RatingPred.csv')

#-----------------------End of BLOCK FIVE - Run as one Set --------------------------------------------------------------------

