import os 
import pandas as pd 

class DataLoader:
    def __init__(self):
        pass

    def load_data(self,data_source):
      if isinstance(data_source,str):

        if not data_source :
           raise ValueError("File Path is empty")
        
        if not data_source.strip():
            raise ValueError("File path contains only spaces")
        
        if not data_source.endswith(".csv"):
            raise ValueError("Only CSV file are allowed")
        
        if not os.path.exists(data_source):
            raise ValueError(f"file not found: {data_source}")
        

        return pd.read_csv(data_source)
      
      else:
         return pd.read_csv(data_source)

        
    def preview_data(self,df):
      return df.head()


if __name__ == "__main__":

       loader=DataLoader()
       df=loader.load_csv("heart.csv")
       print(loader.preview_data(df))
        

        
