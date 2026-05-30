import pandas as pd
import numpy as np


class DatasetAnalyzer:
    def __init__(self,df):
        self.df=df

    def get_shape(self):
    
        return self.df.shape
    
    def get_column_names(self):
        return self.df.columns.tolist()
    
    def get_column_type(self):
        return self.df.dtypes.to_dict()
    
    def missing_values(self):
        return self.df.isna().sum()
    
    def duplicate_count(self):
        return self.df.duplicated().sum()
    
    def numerical_columns(self):
        return self.df.select_dtypes(include=["int64","float64"]).columns.tolist()
    
    def categorical_columns(self):
        return self.df.select_dtypes(include=["object","category"]).columns.tolist()
    
    def generate_report(self):
        report={
            "Shape":self.get_shape(),
            "Columns_name":self.get_column_names(),
            "Columns_type":self.get_column_type(),
            "Missing_values":self.missing_values(),
            "Duplicate_count":self.duplicate_count(),
            "Numerical_columns":self.numerical_columns(),
            "Categorical_columns":self.categorical_columns(),
            
        }

        return report
    


    
