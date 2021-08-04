#Airbnb Price prediction analysis using 
#linear regression,random forest regression, KNN regression and decision tree
#last modified date: 07/25/2021
#require pyspark shell to run the code
#libraries required for data processing
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,r2_score
import math
from pandas import Series,DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#read data into pandas dataframe
df = pd.read_csv("/Users/rmkchowdary/Downloads/AB_US_2020.csv")

#applying trasformations to get curated data
filt = (df.price == 0) | (df.price == 1) 
df = df[~filt]
filt = (df.price > 200) | (df.price <70)
df = df[~filt]
filt = df.availability_365 != 0
df = df[filt]
filt = df.minimum_nights >= 31
df = df[~filt]
df["price"] = df["price"].apply(lambda x: 1 if x < 1 else x)
print(df.isnull().any())
df.reviews_per_month= df.reviews_per_month.fillna(0)
df.last_review = df.last_review.fillna("01/01/01")
df.neighbourhood_group = df.neighbourhood_group.fillna("Other")
df.name = df.name.fillna("AIRBNB HOUSING")
df.host_name = df.host_name.fillna("AIRBNB HOST")
print(df.isnull().any())
#adding state column to input data
cities = list(set(df.city))
df.loc[df.city.isin(cities),'city'].value_counts()
states_dic = {'Asheville':'NC','Austin':'TX','Boston':'MA','Broward County':'FL','Cambridge':'MA','Chicago':'IL','Clark County':'NV','Columbus':'OH','Denver':'CO','Hawaii':'HI','Jersey City':'NJ','Los Angeles':'CA','Nashville':'TN','New Orleans':'LA','New York City':'NY','Oakland':'CA','Pacific Grove':'CA','Portland':'OR','Rhode Island':'RI','Salem':'MA','San Clara Country':'CA','Santa Cruz County':'CA','San Diego':'CA','San Francisco':'CA','San Mateo County':'CA','Seattle':'WA','Twin Cities MSA':'MN','Washington D.C.':'DC'}
states = df.city.apply(lambda x : states_dic[x]) 
df['state'] = df.city.apply(lambda x : states_dic[x]) 
df_lr = df.copy()
df_lr.describe()

# removing outliners 
lower_bound = .25
upper_bound = .75
iqr = df_lr[df_lr['price'].between(df_lr['price'].quantile(lower_bound), df_lr['price'].quantile(upper_bound), inclusive=True)]
iqr = iqr[iqr['number_of_reviews'] > 0]
iqr = iqr[iqr['calculated_host_listings_count'] < 10]
iqr = iqr[iqr['number_of_reviews'] < 200]
iqr = iqr[iqr['minimum_nights'] < 10]
iqr = iqr[iqr['reviews_per_month'] < 5]
df_lr = iqr.copy()
df_lr.describe()

df_x = df_lr.copy()
#set the input parameters to regressor for price prediction
Y = df_x.price
X = df_x.drop('price',axis=1)
#To normalize non numeric labels to numeric import library

encoder = LabelEncoder()

neighbourhood_group = DataFrame({'Neighbourhood_group':X.neighbourhood_group.unique()})
code = encoder.fit_transform(neighbourhood_group['Neighbourhood_group'])
neighbourhood_group['Code'] = code
neighbourhood = DataFrame({'Neighbourhood':X.neighbourhood.unique()})
neigh_code = encoder.fit_transform(neighbourhood['Neighbourhood'])
neighbourhood['Code'] = neigh_code
room_type = DataFrame({'Room type':X.room_type.unique()})
room_code = encoder.fit_transform(room_type['Room type'])
room_type['Code'] = room_code
city = DataFrame({'City' : X.city.unique()})
city_code = encoder.fit_transform(city['City'])
city['Code'] = city_code
state = DataFrame({'state' : X.state.unique()})
state_code = encoder.fit_transform(state['state'])
state['Code'] = state_code
X.neighbourhood_group = encoder.fit_transform(X.neighbourhood_group)
X.neighbourhood = encoder.fit_transform(X.neighbourhood)
X.room_type = encoder.fit_transform(X.room_type)
X.city = encoder.fit_transform(X.city)
X.state = encoder.fit_transform(X.state)
#remove unwanted fields that do not corelate with price
X = X.drop(['id','name','host_id','host_name','latitude','longitude','last_review'],axis=1)
# train the data
x_train,x_test,y_train,y_test = train_test_split(X,Y)

#apply linear regression
reg1 = LinearRegression()
reg1.fit(x_train,y_train)
y_pred_lr = reg1.predict(x_test)
# error analysis for linear regression
print('MAE:', metrics.mean_absolute_error(y_test, y_pred_lr).round(2))
print('MSE:', metrics.mean_squared_error(y_test, y_pred_lr).round(2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr)).round(2))
y_pred_lr = DataFrame({'Actual':y_test,'Predict':y_pred_lr})
y_pred_lr.describe()

# apply Random forest regressor for same trained data

regressor = RandomForestRegressor(n_estimators = 200, random_state = 0)
model=regressor.fit(x_train, y_train)  
y1 = model.predict(x_test)
#error analysis for random forest regressor
print('MAE',metrics.mean_absolute_error(y_test, y1))
print('MSE',mean_squared_error(y_test, y1))
print('RMSE',math.sqrt(mean_squared_error(y_test, y1)))
print('Adj R^2 value:',1 - (1-regressor.score(x_test, y_test))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1))
predictions_rf = pd.DataFrame({"original_Price": y_test.values, "predicted_Price": y1})
predictions_rf.describe()

#apply KNN classifier for same trained data

neigh = KNeighborsClassifier(n_neighbors=3)
scaler = StandardScaler()
scaler.fit(x_train)

X_train = scaler.transform(x_train)
X_test = scaler.transform(x_test)

#fit the model
neigh.fit(X_train, y_train)

# Predicted class
y_pred_knn=neigh.predict(X_test)

# Calculate the accuracy of the model 

print(neigh.score(X_test, y_test)) 
print('Mean absolute Error:', metrics.mean_absolute_error(y_test, y_pred_knn))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_knn))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_knn)))

pred_knn = pd.DataFrame({"original_Price": y_test.values, "predicted_Price": y_pred_knn})
pred_knn.describe()

#apply KNN Regression 

knn = KNeighborsRegressor(algorithm='auto')
# Split into training and test  
X_train, X_test, y_train, y_test = train_test_split( 
             X, Y, test_size = 0.3) 

#fit the model
knn.fit(X_train, y_train)
# Predicted class
y_pred4=knn.predict(X_test)
KNNreg = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred4})
#error analysis
print('Mean absolute Error:', metrics.mean_absolute_error(y_test, y_pred4))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred4)))
KNNreg['squared_error'] = (KNNreg['Predicted'] - KNNreg['Actual'])**(2)
mse = KNNreg['squared_error'].mean()
rmse2 = np.sqrt(mse)
rmse2
KNNreg.describe()

#Apply Decision Tree 
Tree_df=X


#split dataset in features and target variable
feature_cols = ['room_type', 'calculated_host_listings_count']
X = Tree_df[feature_cols].values # Features
#y = Tree_df.price_rng_Cat.values # Target variable

scaler = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X, Y)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train_scaled,y_train)
#Predict the response for test dataset
y_pred_tree = clf.predict(X_test)
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_tree))
#error analysis for decision tree approach
print('Mean absolute Error:', metrics.mean_absolute_error(y_test, y_pred_tree))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_tree))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_tree)))
Treecls = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_tree})
Treecls.describe()

#storing the required data on hdfs for further usage in table creation in hive and visualization in Tableau
#creating a data frame with required fields
df_location = pd.DataFrame({'Listing Id': df.id,'City':df.city,'State': df.state,'longitude':df.longitude,'latitude':df.latitude})
df_listings =pd.DataFrame({'Listing Id': df.id,'Listing Name': df.name})
df_listing_details =pd.DataFrame({'Listing Id': df.id,'Room Type':df.room_type ,'price': df.price, 'No of Reviews': df.number_of_reviews, 'Minimum Nights':df.minimum_nights ,'availability_365': df.availability_365, 'reviews_per_month':df.reviews_per_month})
df_host =pd.DataFrame({'Listing Id': df.id,'Host Id':df.host_id ,'Host Name': df.host_name})

#converting pandas DF to spark DF
df_location = spark.createDataFrame(df_location)
df_listings = spark.createDataFrame(df_listings)
df_listing_details = spark.createDataFrame(df_listing_details)
df_host = spark.createDataFrame(df_host)
pred = spark.createDataFrame(y_pred_lr)
#storing on hdfs  it requires hadoop hdfs to be running otherwise it throws error such as connection refused.
df_location.write.format("csv").save("hdfs://localhost:9000/user/hive/warehouse/location_1", header = True)
df_listings.write.format("csv").save("hdfs://localhost:9000/user/hive/warehouse/listing_1", header = True)
df_listing_details.write.format("csv").save("hdfs://localhost:9000/user/hive/warehouse/listing_details_1", header = True)
df_host.write.format("csv").save("hdfs://localhost:9000/user/hive/warehouse/host_1", header = True)
pred.write.format("csv").save("hdfs://localhost:9000/user/hive/warehouse/prediction_1", header = True)