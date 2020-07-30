import re
from nltk.stem import WordNetLemmatizer
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

class Predictor:
    
    
    def __init__(self):
        "None"
        
        
    def __clean_text(self, rvws):
        
        corpus = []
        for i in range(len(rvws)):
            review = re.sub(r"[^a-zA-Z]", " ", rvws[i])
            review = review.lower().split()
            lemmatizer = WordNetLemmatizer()
            review = [lemmatizer.lemmatize(word) for word in review]
            review = " ".join(review)
            corpus.append(review)
        return corpus
    
    
    def predict(self, reviews, cnt_vec, sentiment_analyzer):
        
        cleaned_text_corpus = self.__clean_text(reviews)
        
        predicted_results = sentiment_analyzer.predict(cnt_vec.transform(cleaned_text_corpus))
        
        output_df = pd.DataFrame({"Reviews":cleaned_text_corpus, 
                                  "sentiment":predicted_results})
        
        pred_prob = pd.DataFrame(sentiment_analyzer.predict_proba(cnt_vec.transform(cleaned_text_corpus)))
        output_df["probabilities"] = -1
        output_df["probabilities"][output_df["sentiment"] == 1] = pred_prob[1][output_df["sentiment"] == 1]
        output_df["probabilities"][output_df["sentiment"] == 0] = pred_prob[0][output_df["sentiment"] == 0]
        
        positive = output_df[output_df["sentiment"] == 1].sort_values(by="probabilities", ascending=False).reset_index(drop=True)
        negative = output_df[output_df["sentiment"] == 0].sort_values(by="probabilities", ascending=False).reset_index(drop=True)
        
        most_positive_review = positive[["Reviews"]][:int(len(positive)*0.3)]
        most_negative_review = negative[["Reviews"]][:int(len(negative)*0.3)]
        
        return most_positive_review, most_negative_review
    
    
if __name__ == "__main__":
    
    "None"