import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import warnings
warnings.filterwarnings("ignore")


df=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")

X=df.drop("Scores",axis=1)
y=df["Scores"]


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
Li_reg=LinearRegression()

Li_reg.fit(X_train,y_train)

pickle.dump(Li_reg,open("model.pkl","wb"))