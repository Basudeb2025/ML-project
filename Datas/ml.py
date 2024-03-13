import pandas as pd
import seaborn as sns
from sklearn import linear_model
df = pd.read_csv(r'C:\Users\Basudeb\Downloads\annual-enterprise-survey-2021-financial-year-provisional-csv.csv')
reg = linear_model.LinearRegression()
reg.predict(df[['Year']],2005)


