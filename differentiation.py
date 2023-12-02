# Differentiation
# 06/03/23

def differentiation(data, rang=1): #rang = range
  diff = list()
  for i in range(rang, len(data)):
    value = data[i] - data[i - rang] #differentiation here
    diff.append(value)
  return Series(diff)
  

# print(series)
differentiated = differentiation(series.values)
# print('////////////////////////////////////')
# print(differentiated)
# pyplot.plot(differentiated)
# pyplot.show()

#Reverse the differentiation

def differentiation_inverse(historic, diff, rang):
  return historic[-rang] + diff
  

#Test
reverse = list()
for i in range(len(differentiated)):
  value = differentiation_inverse(series.values, differentiated[i], len(series) - i)
  reverse.append(value)
reverse = Series(reverse)
print(reverse)


#Arranging the scale
X = series.values
X = X.reshape(len(X), 1)
scale = MinMaxScaler(feature_range = (-1, 1))
scale = scale.fit(X)
X_norm = scale.transform(X)
# X_norm = Series(X_norm) ? #maybe we have an error here
pyplot.plot(X_norm)
pyplot.show()
# value_n = ((value - min) / (max - min)) * 2 - 1
X_orig = scale.inverse_transform(X_norm)
pyplot.plot(X_orig)
pyplot.show()


