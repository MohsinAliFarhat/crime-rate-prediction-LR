import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

df = pd.read_csv('data.csv')
df = df.fillna(1)
sns.heatmap(df.corr())
X = df.drop(columns=['CRIM'])
y = df['CRIM']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression
lin_reg = LinearRegression()
reg = lin_reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)

def mean_squared_error(actual, prediction):
    mse = np.sum(np.square(actual - prediction))
    return mse / len(actual)

print("MSE of testing set: {}".format(MSE(y_test, y_pred)))

# ridge regression
ridge_reg = Ridge(alpha=1,solver='cholesky')
ridge_reg.fit(X_train,y_train)

y_pred = ridge_reg.predict(X_test)

print("MSE of testing set: {}".format(MSE(y_test, y_pred)))