import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

data = pd.read_csv('train.csv')
meal = pd.read_csv('meal_info.csv')
centre = pd.read_csv('fulfilment_center_info.csv')
test = pd.read_csv('test_QoiMO9B.csv')

data.isnull().sum()
meal.isnull().sum()
centre.isnull().sum()

data=pd.merge(data,meal,on='meal_id',how='left',sort=False)

data.isnull().sum()

data.columns

data= data[['id','week','center_id','meal_id','category','cuisine','emailer_for_promotion'
               ,'homepage_featured','checkout_price','base_price','num_orders']]



data=pd.merge(data,centre,on='center_id',how='left',sort=False)


data = data[['id','week','center_id','city_code','region_code','center_type','op_area','meal_id','category','cuisine','emailer_for_promotion'
               ,'homepage_featured','checkout_price','base_price','num_orders']]

test=pd.merge(test,meal,on='meal_id',how='left',sort=False)

test.isnull().sum()

data.columns

test = test[['id','week','center_id','meal_id','category','cuisine','emailer_for_promotion'
               ,'homepage_featured','checkout_price','base_price']]



test=pd.merge(test,centre,on='center_id',how='left',sort=False)


test = test[['id','week','center_id','city_code','region_code','center_type','op_area','meal_id','category','cuisine','emailer_for_promotion'
               ,'homepage_featured','checkout_price','base_price']]



x = data.iloc[:, :-1]
y = data.iloc[:, 14]

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x.iloc[:,5] = labelencoder_x.fit_transform(x.iloc[:,5])
x.iloc[:,8] = labelencoder_x.fit_transform(x.iloc[:,8])
x.iloc[:,9] = labelencoder_x.fit_transform(x.iloc[:,9])

X1=pd.DataFrame(x)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X1)
X_train=pd.DataFrame(X_train)

X_train1 = pd.DataFrame(X_train)

X_train1.columns=x.columns
X_train=pd.DataFrame(X_train1)


Y_train=y
Y_train=pd.DataFrame(Y_train)


x_test = test.iloc[:, :14]


from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x_test.iloc[:,5] = labelencoder_x.fit_transform(x_test.iloc[:,5])
x_test.iloc[:,8] = labelencoder_x.fit_transform(x_test.iloc[:,8])
x_test.iloc[:,9] = labelencoder_x.fit_transform(x_test.iloc[:,9])

X4=pd.DataFrame(x_test)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_test = sc_X.fit_transform(X4)
X_test=pd.DataFrame(X_test)
X_test.columns=x_test.columns

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)



# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))




























