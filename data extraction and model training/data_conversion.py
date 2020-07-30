import pandas as pd
from bs4 import BeautifulSoup
import os
import warnings

warnings.filterwarnings("ignore")


label = pd.DataFrame(columns=["reviews", "class"])

path = os.getcwd()

files = os.listdir(path+"\\sorted_data")

for f in files:
    
    if os.path.isdir(path+"\\sorted_data"+"\\{}".format(f)):
        
        for lbl in ["positive", "negative"]:
            
            xml = path+"\\sorted_data"+"\\{}".format(f)+"\\{}.review".format(lbl)
            with open(xml, 'r') as infile:
                contents = infile.read()
                infile.close()

            contents = contents.split("</review>")
            contents = "</review>^*".join(contents)
            contents = contents.split("^*")[:-1]
            
            if lbl == "positive":
                value = 1
            else:
                value = 0

            for d in contents:
                soup = BeautifulSoup(d, 'xml')
                review = soup.find("review_text")
                res = {"reviews":"{}".format(review.getText()), "class":value}
                label = label.append(res, ignore_index=True)

label.to_csv("product_reviews.tsv", sep = '\t', index=False)

