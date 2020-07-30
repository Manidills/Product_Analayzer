# Natural Language Processing

# Importing the libraries
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import warnings

warnings.filterwarnings("ignore")


class SentimentAnalyzer:
    
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
        self.cnt_vec = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        
    def __clean_text(self):
        
        print("text cleaning....")
        for i in range(len(self.X)):
            review = re.sub(r"[^a-zA-Z]", " ", self.X[i])
            review = review.lower().split()
            lemmatizer = WordNetLemmatizer()
            review = [lemmatizer.lemmatize(word) for word in review]
            review = " ".join(review)
            self.X[i] = review
        print("text cleaning complete")
            

    def __feature_engineering(self):
        
        self.cnt_vec = CountVectorizer(tokenizer=word_tokenize,
                                         token_pattern=None,
                                         ngram_range=(1, 2))
        self.cnt_vec.fit(self.X)

        train = self.cnt_vec.transform(self.X)
        print("training...")
        
        return train
        
    
    def train(self):
        
        self.__clean_text()

        self.X_train = self.__feature_engineering()
        
        classifier = linear_model.LogisticRegression(random_state=2020)
        
        classifier.fit(self.X_train, self.y)
        
        return classifier, self.cnt_vec
    

    def k_fold(self):
        
        df = pd.concat([self.X, self.y], axis=1)

        #creating a new column and filling it with -1
        df["k-fold"] = -1

        # getting labels
        y = df["class"].values

        # initializing the kfold class from model_selection module with 5 folds
        kf = StratifiedKFold(n_splits=5)

        # filling K-fold column values assigned to test data
        for num, (train_idx, test_idx) in enumerate(kf.split(X=df, y=y)):
            df.loc[test_idx, "K-fold"] = num
    
    
        # Iterating the folds created
        for fold_ in range(5):
    
            # temporary dataframes for train and test
            train_df = df[df["K-fold"] != fold_].reset_index(drop=True)
            test_df = df[df["K-fold"] == fold_].reset_index(drop=True)
            
            # transform training and validation data reviews
            xtrain = self.cnt_vec.transform(train_df["reviews"])
            xtest = self.cnt_vec.transform(test_df["reviews"])
    
            # initialize logistic regression model
            classifier = linear_model.LogisticRegression(random_state=2020)
            #classifier = naive_bayes.MultinomialNB()
    
            # training the model
            classifier.fit(xtrain, train_df["class"])
    
            # make predictions on test data
            # threshold for predictions is 0.5
            preds = classifier.predict(xtest)
    
            # calculate accuracy
            accuracy = metrics.accuracy_score(test_df["class"], preds)
    
            print(f"Fold: {fold_}")
            print(f"Accuracy = {accuracy}")
            print("")


def main():
    # Importing the dataset
    df = pd.read_csv('product_reviews_balanced.tsv', delimiter = '\t')
    print(df["class"].value_counts())
    
    # randomizing the rows of the data
    df = df.sample(frac=1, random_state=2020).reset_index(drop=True)
    
    X = df["reviews"] 
    y = df["class"]
    
    learner = SentimentAnalyzer(X=X, y=y)
    
    classifier, cnt_vec = learner.train()
    
    learner.k_fold()
    
    return classifier, cnt_vec
    

if __name__ == "__main__":
    
    trained_model, cnt_vec = main()

    joblib_file1 = "SentimentAnalyzer.pkl" 
    joblib.dump(trained_model, joblib_file1)
    
    joblib_file2 = "CountVectorizer.pkl" 
    joblib.dump(cnt_vec, joblib_file2)