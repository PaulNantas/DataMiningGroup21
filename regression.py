from os import X_OK
from pandas.io.parsers import read_csv
from create_dataset import *
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

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

#Function that return DataFrame with id corresponding to a continent and 
#a dict continent_id to know which id correspond to which continent
def continent_to_id(df):
  continent = df['ParentLocation']
  continent_id = {'Eastern Mediterranean':0, 'Europe':1, 'Africa':2 ,'Americas':3, 'Western Pacific':4,'South-East Asia':5 } 
  df["continent_id"] = continent.map(continent_id)

  return(df,continent_id)

#Function returns datadframe with the sexe selected and the year. if flag_year = False, we don't care about years
def choose_year_sexe(df,sexe_choice,flag_year,year=2014): #sexe_choice = 'MLE','FMLE','BTSX'
  df_new = df.loc[df['Dim1ValueCode'] == sexe_choice]
  df_new = df_new.drop(['Dim1ValueCode'], axis=1)
  
  if flag_year:
    df_new = df_new.loc[df_new['Period'] == year ]
    df_new = df_new.drop(['Period'], axis=1)
  #else:
  #  df_new = df_new.drop(columns='Period',axis=1)
  
  return(df_new)

#Return a dataframe with for each data those column
def order_data(df,order):
    features = ['AR','MBMI','FPM','ABR']
    for k in range(2,order+1):
      for item in features:
        df[f"Power {k} of {item}"] = np.power(df[item],k)

    return (df)

#Split data into test and train set ; flag_squared : If you want 2nd order regression
def data_split(df,sexe_choice,order,flag_year):
    X = df.drop(['SR'],axis=1)
    if sexe_choice == 'MLE':
      X = X.drop(['ABR'], axis=1)
      if order >1:
        for k in range(2,order+1):
          X = X.drop([f'Power {k} of ABR'], axis=1)

    if flag_year:
      y = df[['SR']]
      X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.32)#,random_state=42)
    else:
      y = df[['Period','SR']]

      #2014 : 111, 2015 : 105, 2016 : 112
      #X_train : 2014,2016 -> 223 68%
      #X_test : 2015 -> 105 32%
      X_train  = X[(X['Period'] == 2014) | (X['Period'] == 2016)]
      y_train = y[(y['Period'] == 2014) | (y['Period'] == 2016)]
      X_test  = X[(X['Period'] == 2015)]
      y_test = y[(y['Period'] == 2015)]

      y_train = y_train.drop(['Period'], axis=1)
      y_test = y_test.drop(['Period'], axis=1)
    return (X_train,X_test,y_train,y_test)

#Normalize data
def normalization(df):

    names = df.columns
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    df = df.set_axis(names, axis=1)#, inplace=False))

    return df

#Regression function : return the prediction 
def regression(X_train,X_test,y_train,y_test,flag_country,flag_continent):
    if flag_country:
      X_train = X_train.loc[:, X_train.columns != 'country_id']
      X_test = X_test.loc[:, X_test.columns != 'country_id']
      y_train = y_train.loc[:, y_train.columns != 'country_id']
      y_test = y_test.loc[:, y_test.columns != 'country_id']

    if flag_continent:
      X_train = X_train.loc[:, X_train.columns != 'continent_id']
      X_test = X_test.loc[:, X_test.columns != 'continent_id']
      y_train = y_train.loc[:, y_train.columns != 'continent_id']
      y_test = y_test.loc[:, y_test.columns != 'continent_id']


    LR = LinearRegression()
    LR.fit(X_train,y_train)
    y_predict = LR.predict(X_test)
    y_test = y_test.to_numpy()


    score=r2_score(y_test,y_predict)
    MSE = mean_squared_error(y_test,y_predict)
    #print("Score of regression")
    #print('R2 =',score)
    #print('MSE =',MSE)
    #print('RMSE =',np.sqrt(MSE))

    error = np.abs(y_test-y_predict)
    
    return (y_predict,y_test,error,np.sqrt(MSE))

#Return the list of countries in y_test
def countries(y_test,id):
    y_test = y_test.to_numpy()
    list_countries = []
    #for c in y_test[:,0]:
    for c in y_test:
        list_countries.append(id[int(c)][0])

    return(list_countries)

if __name__=='__main__':
    #df = merge_data_sets()
    df = read_csv("data/data_set.csv")
    df_regression, countries_id  = country_to_id(df)
    df_regression, continents_id  = continent_to_id(df)

    print(df_regression)
    print('Suicide rate parameters')
    print('Mean =',df_regression['SR'].mean())
    print('Std =',df_regression['SR'].std())
  
    ############ FLAG SELECTION ##############
    sexe_choice = 'BTSX' #sexe_choice = 'MLE','FMLE','BTSX'
    flag_year = False #False -> Train : 2014,2016, Test -> 2015 and True -> year is by default 2014 and split of 0.75/0.25 train/test
    flag_squared = True #Add squared data to dataframe
    flag_country = True # If you don't want the id of the country in the regression 
    flag_continent = True # If you don't want the id of the continent in the regression 
    order = 4 #Order of the polynomial regression 

    ############# DATA_TO_KEEP_FOR_REGRESSION#############
    data_name = ['country_id','continent_id',"Period","Dim1ValueCode",'AR','MBMI','FPM','ABR','SR']
    
    gender = ['MLE','FMLE','BTSX']
    predictions = []
    for sexe in gender : 
      df_regression = df_regression[data_name]
      #print(df_regression.describe())
      df = df_regression.copy()
      df = choose_year_sexe(df,sexe,flag_year,year=2014)
      df = order_data(df,order)
      

      X_train,X_test,y_train,y_test = data_split(df,sexe,order, flag_year)
      list_countries = countries(X_test['country_id'],countries_id)
      X_train,X_test = normalization(X_train),normalization(X_test)
      print('STATISTICAL DATA')
      print(f'Mean y_test {sexe}', np.round(float(y_test.mean()),2))
      print(f'Std y_test {sexe}', np.round(float(y_test.std()),2))
      y_predict,y_test,error,RMSE = regression(X_train,X_test,y_train,y_test,flag_country,flag_continent)
      predictions.append([y_predict,y_test, error,RMSE])


    plt.plot(list_countries,predictions[0][2],color='b',label='Male')
    plt.plot(list_countries,predictions[1][2],color='r',label='Female')
    plt.plot(list_countries,predictions[2][2],color='g',label='Both Genders')
    plt.axhline(y=predictions[0][3], color='b', linestyle='dashed',label = f'RMSE for Male {np.round(predictions[0][3],4)}')
    plt.axhline(y=predictions[1][3], color='r', linestyle='dashed',label=f'RMSE for Female {np.round(predictions[1][3],4)}')
    plt.axhline(y=predictions[2][3], color='g', linestyle='dashed',label=f'RMSE for Both Genders {np.round(predictions[2][3],4)}')

    plt.title('Prediction error  of the suicide rate per 100,000 inhabitants')
    plt.xticks(rotation=60)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xlabel('Countries')
    plt.ylabel('Error |y_test-y_predict|')
    plt.legend(prop={"size":15})
    plt.show()
    
    plt.show(block=False)
    plt.pause(20)
    plt.close() 
    