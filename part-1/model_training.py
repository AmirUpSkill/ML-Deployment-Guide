# model_training.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib 
from config import TEST_SIZE, RANDOM_STATE

def split_data(X, y):
    """ 
    Splits the data into train and test sets
    """
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
def train_model(X_train, y_train, model_type='logistic_regression'):
    """Trains and returns a model."""
    if model_type == 'logistic_regression':
        model = LogisticRegression()
    elif model_type == 'knn':
        model = KNeighborsClassifier()
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier()
    else:
        raise ValueError("Invalid model type")
    
    model.fit(X_train, y_train)
    return model
def create_pipeline(model_type='logistic_regression'):
    """Create a pipeline with StandardScaler and a chosen classifier."""
    if model_type == 'logistic_regression':
        model = LogisticRegression()
    elif model_type == 'knn':
        model = KNeighborsClassifier()
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier()
    else:
        raise ValueError("Invalid model type")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    return pipeline
def evaluate_model(y_test, y_pred):
    """Evaluates the model and prints classification report and confusion matrix."""
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    return classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred)

def save_model(model, filename):
    """Saves the trained model to a file."""
    joblib.dump(model, filename)

def load_model(filename):
    """Loads the trained model from a file."""
    return joblib.load(filename)