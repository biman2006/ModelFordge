from analysis.dataset_analyzer import DatasetAnalyzer
import pandas as pd


print("PREPROCESSOR FILE LOADED")
class Preprocessor:
    def __init__(self,df):
        print("ENCODING STRATEGY RECEIVED")
        
        print("PREPROCESSOR INIT CALLED")
        self.df=df
        
    
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
    


    
    def handle_missing_values(self,numeric_strategy):

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
                    if numeric_strategy=="mean":
                        fill_value=df_copy[column].mean()

                    elif numeric_strategy=="median":
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
    

    

    def handle_duplicate(self,duplicate_strategy):

        df_copy=self.df.copy()
        if self.analyzer.duplicate_count()==0:
            return df_copy
        else:
            if duplicate_strategy=="keep":
                return df_copy
            elif duplicate_strategy=="remove":
                df_copy=df_copy.drop_duplicates()

                return df_copy
            


            
    def encode_categorical_column(self,column_name, encoding_strategy):

        df_copy=self.df.copy()

        
        

        if not column_name:
            return df_copy

        if encoding_strategy=="keep_Same":
            return df_copy 
        
        if encoding_strategy=="OneHotEncoder":
          
          from sklearn.preprocessing import OneHotEncoder

          ohe=OneHotEncoder(sparse_output=False,handle_unknown='ignore')

          encoded_data=ohe.fit_transform(df_copy[[column_name]])

          encoded_df=pd.DataFrame(encoded_data,columns=ohe.get_feature_names_out([column_name]), index=df_copy.index)

          df_copy=df_copy.drop(columns=column_name)

          df_copy=pd.concat([df_copy,encoded_df],axis=1)
        

          return df_copy
        

        elif encoding_strategy=="LabelEncoder":
            from sklearn.preprocessing import LabelEncoder

            

            Le=LabelEncoder()

            df_copy[column_name]=Le.fit_transform(df_copy[column_name].astype(str))

            return df_copy
        

        elif encoding_strategy=="OrdinalEncoder":
          
          from sklearn.preprocessing import OrdinalEncoder

          ore=OrdinalEncoder()

          encoded_data=ore.fit_transform(df_copy[[column_name]])

          
          df_copy[column_name]=encoded_data.flatten()

          return df_copy
        

        
    def detect_problem_type(self,target_column):

        df_copy=self.df.copy()
        y=df_copy[target_column]

        unique_values=y.nunique()

        if y.dtype=="object" or y.dtype=="category":
            return "Classification"
        
        elif unique_values<=15:
            return "Classification"
        
        else:
            return "Regression"


        
        
    def train_test_split_data(self,target_column,test_size=0.2,random_state=42):
        df_copy=self.df.copy()

        X=df_copy.drop(columns=target_column)
        y=df_copy[target_column]

        problem_type=self.detect_problem_type(target_column)
        

        from sklearn.model_selection import train_test_split

        if problem_type=="Classification":

          X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=y)
        else:
              X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=random_state)


        return (X_train,X_test,y_train,y_test)
    

    
    def scale_column(self,X_train,X_test,column_name,scaling_strategy):
        df_copy=self.df.copy()

    

        
        if not pd.api.types.is_numeric_dtype(
              X_train[column_name]
                      ) :
                raise ValueError(
        "Scaling applies only to numeric columns"
    )
        
        else:
            if scaling_strategy=="StandardScaler":
                from sklearn.preprocessing import StandardScaler

                scaler=StandardScaler()

                

                
            
            elif scaling_strategy=="MinMaxScaler":
                from sklearn.preprocessing import MinMaxScaler

                scaler=MinMaxScaler()

            elif scaling_strategy=="RobustScaler":
                from sklearn.preprocessing import RobustScaler
                scaler=RobustScaler()

            elif scaling_strategy=="Keep Same":
                return X_train,X_test
            else:
                raise ValueError("Invalid Scaling Strategy")
            


            X_train[column_name]=scaler.fit_transform(X_train[[column_name]])

            X_test[column_name]=scaler.transform(X_test[[column_name]])

            return X_train,X_test
                

    

        

             

            

             
        

          
        



            





        




