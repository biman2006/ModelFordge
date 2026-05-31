from analysis.dataset_analyzer import DatasetAnalyzer
import pandas as pd

print("PREPROCESSOR FILE LOADED")
class Preprocessor:
    def __init__(self,df,numeric_strategy,cat_strategy,duplicate_strategy):
        print("PREPROCESSOR INIT CALLED")
        self.df=df 
        self.numeric_strategy=numeric_strategy
        self.cat_strategy=cat_strategy
        self.duplicate_strategy=duplicate_strategy
        self.analyzer=DatasetAnalyzer(self.df)


    def get_missing_columns(self):
        missing_value_cols=self.df.columns[self.analyzer.missing_values()>0].tolist()

        return missing_value_cols
    
    def calculate_missing_percentage(self):
        missing_counts=self.analyzer.missing_values()
        missing_counts=missing_counts[missing_counts>0]
        result=pd.DataFrame({
            'column':missing_counts.index,
            'Missing_value_percentage':(missing_counts)/(self.df.shape[0])*100
        })

        return result
    
    def handle_missing_values(self):

        df_copy=self.df.copy()

        missing_info=self.calculate_missing_percentage()

        numeric_columns=self.analyzer.numerical_columns()
        categorical_columns=self.analyzer.categorical_columns()

        high_missing_columns=[]

        for _,row in missing_info.iterrows():
            column=row["column"]

            missing_percentage=row["Missing_value_percentage"]

            if missing_percentage==0:
              continue

            elif missing_percentage<=5:
                if column in numeric_columns:
                    if self.numeric_strategy=="mean":
                        fill_value=df_copy[column].mean()

                    elif self.numeric_strategy=="median":
                        fill_value=df_copy[column].median()
                    df_copy[column]=df_copy[column].fillna(fill_value)

                elif column in categorical_columns:
                    fill_value=df_copy[column].mode()[0]

                    df_copy[column]=df_copy[column].fillna(fill_value)

            elif missing_percentage>40:
                high_missing_columns.append(column)

            else:
                if column in numeric_columns:
                    fill_value=df_copy[column].median()
                    df_copy[column]=df_copy[column].fillna(fill_value)

                elif column in categorical_columns:
                    fill_value=df_copy[column].mode()[0]
                    df_copy[column]=df_copy[column].fillna(fill_value)


        return df_copy,high_missing_columns
    

    def handle_duplicate(self):

        df_copy=self.df.copy()
        if self.analyzer.duplicate_count()==0:
            return df_copy
        else:
            if self.duplicate_strategy=="keep":
                return df_copy
            elif self.duplicate_strategy=="remove":
                df_copy=df_copy.drop_duplicates()

                return df_copy
            





        




