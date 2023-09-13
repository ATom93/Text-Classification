import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, LearningCurveDisplay, \
     GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC
from wordcloud import WordCloud


def write_on_file(filename, append, data):
    if append:
        mode = "a"
    else:
        mode = "w"
    f = open(filename, mode)
    f.write(data)
    f.close()

def read_data():
    # Read the CSV data from the provided URL
    data = pd.read_csv('https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv', encoding='latin-1')
    # Drop unnecessary columns (axis) from the dataset
    columns_to_drop = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
    data.drop(
        columns_to_drop,
        axis=1,
        inplace=True
    )
    # Rename the remaining columns as 'label' and 'text'
    data.columns = ['label', 'text']
    # Print the first few rows of the data
    #print("\nData:\n", data.head())
    print("Number of entries:\t", data.shape[0])
    return data

def check_data(data):
    # Calculate the counts of unique values in the 'label' column
    label_counts = data['label'].value_counts()
    # and plot them as a pie chart.
    label_counts.plot(
        kind = 'pie',                       # Set the chart type to 'pie'.
        explode = [0, 0.1],                 # 'explode' parameter separates the second slice from the center to emphasize it (0.1 means 10% offset).
        figsize = (6, 6),                   # Set the size of the pie chart to 6x6 inches.
        autopct = '%1.1f%%',                # Show percentage values with one decimal place on each slice of the pie.
        shadow = True                       # Display a shadow effect behind the pie chart.
    )
    plt.ylabel("Spam vs Ham")
    plt.legend(["Ham", "Spam"])
    plt.show()

def text_preprocessing(data):
    # Download NLTK resources (if needed)
    nltk.download('all')
    # Extract the 'text' column from the given data in a list
    text = list(data['text'])
    # Initialize the WordNetLemmatizer for lemmatization
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Initialize an empty list to store the preprocessed texts
    corpus = []
    # Perform text preprocessing on each text
    for i in range(len(text)):
        # Remove non-alphabetic characters and replace them with a space
        r = re.sub('[^a-zA-Z]', ' ', text[i])
        # Convert the text to lowercase
        r = r.lower()
        # Split the text into individual words
        r = r.split()
        # Remove stopwords from the text
        r = [word                                           #3
             for word in r                                  #1
             if word not in stop_words]     #2
        # Lemmatize the words in the text
        r = [lemmatizer.lemmatize(word)
             for word in r]
        # Join the preprocessed words back into a single text
        r = ' '.join(r)
        # Append the preprocessed text to the corpus
        corpus.append(r)
    data['text'] = corpus
    # Update the 'text' column in the data with the preprocessed texts
    return data

def feature_extraction(train_test):
    # Create a CountVectorizer object
    cv = CountVectorizer()
    # Perform feature extraction on the training data
    X_train_cv = cv.fit_transform(train_test["X_train"])
    #print("\nTokens:\t", type(cv.get_feature_names_out()))
    pd.DataFrame(cv.get_feature_names_out()).to_csv("data/tokens.csv")
    #print("\nMatrix of token counts:\n", X_train_cv.toarray())
    pd.DataFrame(X_train_cv.toarray()).to_csv("data/tokens_matrix.csv")
    #print("\nShape of Matrix of token counts:\t", type(X_train_cv.shape))
    # Perform feature extraction on the test data using the same vocabulary
    X_test_cv = cv.transform(train_test["X_test"])
    return {
        "X_train_cv": X_train_cv,
        "X_test_cv": X_test_cv
    }
    #return (X_train_cv, X_test_cv)


##MODEL
def train_test(data):
    X = data['text']
    Y = data['label']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=123)

    print('Training Data:\t', X_train.shape)
    #print('\tSpam in training data:\t' + str(Y_train.value_counts()['spam']))
    #print('\tHam in training data:\t' + str(Y_train.value_counts()['ham']))
    print('Testing data: ', X_test.shape)
    #print('\tSpam in training data:\t' + str(Y_test.value_counts()['spam']))
    #print('\tHam in training data:\t' + str(Y_test.value_counts()['ham']))

    pd.DataFrame({'text': X_train, 'label': Y_train}).to_csv("data/training_data.csv", index=False)
    pd.DataFrame({'text': X_test, 'label': Y_test}).to_csv("data/test_data.csv", index=False)

    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_test": X_test,
        "Y_test": Y_test
    }
    #return (X_train, X_test, Y_train, Y_test)

def model_training(model, X_train_cv, y_train):
    #print("\n---------------", type(model).__name__, "---------------\n")
    start = time.time()
    model.fit(X_train_cv, y_train)
    stop = time.time()
    print(f"Training time: {stop - start}s")
    return model

def generate_predictions(model, data):
    return model.predict(data)

def compute_confusion_matrix(y_test, predictions):
    confusion_matrix = metrics.confusion_matrix(y_test, predictions)
    print("Performance report\n",
          metrics.classification_report(y_test, predictions))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    disp.plot()
    plt.show()
    # return pd.DataFrame(
    #     confusion_matrix,
    #     index=['ham', 'spam'],
    #     columns=['ham', 'spam']
    # )


def perform_grid_search(model, param_grid, model_name):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X_cv['X_train_cv'], train_test["Y_train"])
    print("%s\t best: %f using %s" % (model_name, grid_result.best_score_, grid_result.best_params_))
    return grid_result

def get_tuned_LR():
    model = LogisticRegression()
    param_grid = {
        'penalty': ['l2'],
        'C': np.logspace(-4, 4, 20),
        'solver': ['lbfgs']
    }
    model = perform_grid_search(model, param_grid, "LOGISTIC REGRESSION")
    return model

def get_tuned_SVC():
    model = SVC()
    param_grid = {
        'kernel': ['poly', 'rbf', 'sigmoid'],
        'C': [50, 10, 1.0, 0.1, 0.01],
        'gamma': ['scale']
    }
    model = perform_grid_search(model, param_grid, "SVC")
    return model

def get_tuned_RFC():
    model = RandomForestClassifier()
    param_grid = {
        'n_estimators': [25, 50, 100, 150],
        'max_features': ['sqrt', 'log2'],#, None],
        'max_depth': [3, 6, 9],
        'max_leaf_nodes': [3, 6, 9],
    }
    model = perform_grid_search(model, param_grid, "RANDOM FOREST")
    return model

if __name__ == '__main__':
    data = read_data()
    check_data(data)

    #word_analysis(data)

    data = text_preprocessing(data)
    train_test = train_test(data)
    X_cv = feature_extraction(train_test)


    models = [
        get_tuned_LR(),
        get_tuned_SVC(),
        get_tuned_RFC()
    ]


    for model in models:
        predictions = generate_predictions(model, X_cv["X_test_cv"])
        print("Accuracy on test set:\t", accuracy_score(train_test["Y_test"], predictions))
        compute_confusion_matrix(train_test["Y_test"], predictions)


    #plot_learning_curve(X_cv["X_train_cv"], train_test["Y_train"], models)
