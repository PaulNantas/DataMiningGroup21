from create_dataset import *

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib
import matplotlib.pyplot as plt


#Function that return DataFrame with id corresponding to a country and 
#a list countries_id to know which id correspond to which country
def country_to_id(df):
  countries = df['Location']
  nc = 0
  country = countries[0]
  id = []
  countries_id = [[countries[0],0]]

  for k in countries:
    if country == k:
      id.append(nc)
    else : 
      nc+=1
      country = k
      id.append(nc)
      countries_id.append([k,nc])
  df["country_id"] = id

  return(df,countries_id)

#Function returns datadframe with the sex selected and the year. if flag_year = False, we don't care about years
def choose_year_sex(df,sex_choice,flag_year,year=2014): #sex_choice = 'MLE','FMLE','BTSX'
  df_new = df.loc[df['Dim1ValueCode'] == sex_choice]
  df_new = df_new.drop(['Dim1ValueCode'], axis=1)

  if flag_year:
    df_new = df_new.loc[df_new['Period'] == year ]
    df_new = df_new.drop(['Period'], axis=1)
  else:
    df_new = df_new.drop(columns='Period',axis=1)

  return(df_new)

#Return a DataFrame with squarred data added
def squared_data(df):
    df['Squared AR'] = np.square(df['AR'])
    df['Squared MBMI'] = np.square(df['MBMI'])
    df['Squared FPM'] = np.square(df['FPM'])
    df['Squared ABR'] = np.square(df['ABR'])

    return (df)

#Split data into test and train set ; flag_squared : If you want 2nd order regression

def data_split(df,sex_choice,flag_squared):
    if sex_choice == 'MLE':
        if flag_squared:
            X = df[['country_id','AR','MBMI','FPM','Squared AR','Squared MBMI','Squared FPM']]
        else:
            X = df[['country_id','AR','MBMI','FPM']]    
    else:
        if flag_squared:
            X = df[['country_id','AR','MBMI','FPM','ABR','Squared AR','Squared MBMI','Squared FPM','Squared ABR']]
        else:
            X = df[['country_id','AR','MBMI','FPM','ABR']]

    y = df[['country_id','SR']]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)#,random_state=2)

    return (X_train,X_test,y_train,y_test)

#Regression function : return the prediction 
def regression(X_train,X_test,y_train,y_test):
    LR= LinearRegression()
    LR.fit(X_train,y_train)
    y_predict = LR.predict(X_test)

    score=r2_score(y_test,y_predict)
    print('R2 =',score)
    print('MSE =',mean_squared_error(y_test,y_predict))
    print('RMSE =',np.sqrt(mean_squared_error(y_test,y_predict)))

    return (y_predict)

#Return the list of countries in y_test
def countries(y_test,id):
    y_test = y_test.to_numpy()
    list_country = []
    for c in y_test[:,0]:
        list_country.append(id[int(c)][0])

    return(list_country)

if __name__=='__main__':
    df = merge_data_sets()

    df_regression, countries_id  = country_to_id(df)

    data_name = ['country_id',"Period","Dim1ValueCode",'AR','MBMI','FPM','ABR','SR']
    df_regression = df_regression[data_name]

    sex_choice = 'BTSX' #sex_choice = 'MLE','FMLE','BTSX'
    flag_year = True #Want a specific year, by default is 2014
    flag_squared = False #Add squared data to dataframe
    df_regression = choose_year_sex(df_regression,sex_choice,flag_year)

    if flag_squared:
        df_regression = squared_data(df_regression)

    X_train,X_test,y_train,y_test = data_split(df_regression,sex_choice,flag_squared)
    y_predict = regression(X_train,X_test,y_train,y_test)
    list_countries = countries(y_test,countries_id)
    print(list_countries)

    plt.scatter(list_countries,y_test.to_numpy()[:,1]-y_predict[:,1],color='b')
    plt.title('Prediction error of the suicide rate per 100,000 inhabitants in absolute value')
    plt.xticks(rotation=60)
    plt.tick_params(axis='both', which='major', labelsize=8)

    plt.show()