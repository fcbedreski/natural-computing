# Recurrent Neural Network - RNN
# 27/02/23


#CSV file origin
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv'


#Libraries
from pandas import read_csv, datetime, DataFrame, concat, Series
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler

#Arranging the date
def parser(x):
  return datetime.strptime('190'+x, '%Y-%m')
  
#"Other addenda"
def timeSeries_to_supervised(data, lag = 1):
  df = DataFrame(data)
  columns = [df.shift(i) for i in range(1, lag + 1)]
  columns.append(df)
  df = concat(columns, axis = 1)
  df.fillna(0, inplace = True)
  return df
  

#Reading data
series = read_csv(url, header = 0, parse_dates = [0], index_col = 0, squeeze = True, date_parser = parser)

#Verifying
print(series.head())
series.plot()
pyplot.show()


X = series.values
supervised = timeSeries_to_supervised(X, 1)
print(supervised.head())

