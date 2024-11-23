import pandas as pd 
from sklearn.datasets import load_iris

def load_data():
    """ 
    Loads the iris dataset from sklearn
    Returns:
       data: pd.DataFrame
    """
    iris=load_iris()
    data=pd.DataFrame(iris.data,columns=iris.feature_names)
    data['target']=iris.target_names[iris.target]
    return data 
