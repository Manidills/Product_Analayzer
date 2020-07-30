import requests
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings("ignore")


class Scrap:
    
    header = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36"}
    
    
    def __init__(self, url):
        
        self.url = url
    
    
    def __amazon(self):
        
        a = self.url.split("/dp/")
        b = a[0] + "/product-reviews/" + a[1].split("/")[0]
        
        reviews = []
        
        for page in range(1, 5):
            
            reviews_url = b + "/ref=cm_cr_arp_d_paging_btm_next_{i}?ie=UTF8&reviewerType=all_reviews&pageNumber={i}".format(i=page)
            reviews_page = requests.get(reviews_url, headers=Scrap.header)
            
            soup = BeautifulSoup(reviews_page.content, features="lxml")
            scraped_data = soup.findAll("span", {"data-hook":"review-body"})
            
            if scraped_data != []:
                reviews = reviews + [review.text for review in scraped_data]
            else:
                break
            
        return reviews
    
    
    def __flipkart(self):
        
        a = self.url.split("/p/")
        b = a[0] + "/product-reviews/" + a[1].split("&srno")[0]
        
        reviews = []
        
        for page in range(1, 5):
            
            reviews_url = b + "&page={i}".format(i=page)
            reviews_page = requests.get(reviews_url, headers=Scrap.header)
            
            soup = BeautifulSoup(reviews_page.content, features="lxml")
            scraped_data = soup.findAll("div", {"class":"qwjRop"})
            
            if scraped_data != []:
                reviews = reviews + [review.text for review in scraped_data]
            else:
                break
            
        return reviews


    def scrap(self):
        
        rvws = None
        
        if "www.amazon" in self.url[:25]:
            rvws = self.__amazon()
            
        elif "www.flipkart" in self.url[:25]:
            rvws = self.__flipkart()
        
        return rvws


if __name__ == "__main__":
    None
