class ModelTrainer:


    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier 
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        from sklearn.naive_bayes import GaussianNB


        self.CLASSIFICATION_MODELS={
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Support Vector Machine": SVC(),
            "Neural Network":MLPClassifier(),
            "Naive Bayes":GaussianNB()
        }

        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        from sklearn.linear_model import Lasso
        from sklearn.linear_model import Ridge

        self.REGRESSION_MODELS={
            "Linear Regression": LinearRegression(),
            "Lasso Regression": Lasso(),
            "Ridge Regression": Ridge(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Support Vector Machine": SVR(),
            "Decision Tree Regressor": DecisionTreeRegressor()
        }

    def train_model(self,X_train,y_train,algorithm):

        

        if algorithm in self.CLASSIFICATION_MODELS:
            model=self.CLASSIFICATION_MODELS[algorithm]

        elif algorithm in self.REGRESSION_MODELS:
            model=self.REGRESSION_MODELS[algorithm]

        else:
            raise ValueError("Invalid Algorithm")
        
        model.fit(X_train,y_train)

        return model
    

    def evaluate_model(self,model,X_test,y_test,problem_type):
        y_pred=model.predict(X_test)


        if problem_type=="Classification":
            
            from sklearn.metrics import confusion_matrix 
            from sklearn.metrics import classification_report
            from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score

            CM=confusion_matrix(y_test,y_pred)
            CR=classification_report(y_test,y_pred)
            accuracy=accuracy_score(y_test,y_pred)
            precision=precision_score(y_test,y_pred,average="weighted")
            f1=f1_score(y_test,y_pred,average="weighted")

            

            return {
                "Accuracy":accuracy,
                "Precision":precision,
                "Recall":recall_score(y_test,y_pred,average="weighted"),
                "F1_score":f1,

                "Confusion_Matrix":CM,
                "Classification_Report":CR}
        

        elif problem_type=="Regression":
            from sklearn.metrics import mean_absolute_error,root_mean_squared_error,mean_squared_error,r2_score

            MAE=mean_absolute_error(y_test,y_pred)
            MSE=mean_squared_error(y_test,y_pred)
            RMSE=root_mean_squared_error(y_test,y_pred)

            R2_Score=r2_score(y_test,y_pred)

            return {
                "Mean Absolute Error":MAE,
                "Mean Squared Error":MSE,
                "Root Mean Squared Error":RMSE,
                "R2_Score":R2_Score
            }


    def compare_models(self,X_train,y_train,X_test,y_test,problem_type):
        
        import pandas as pd

        if problem_type=="Classification":
            models=self.CLASSIFICATION_MODELS
        else:
            models=self.REGRESSION_MODELS

        results=[]

        for model_name,model in models.items():
            trained_model=self.train_model(X_train,y_train,model_name)
            metrics=self.evaluate_model(trained_model,X_test,y_test,problem_type)

            if problem_type=="Classification":
                results.append({"Model":model_name,
                                "Accuracy":metrics["Accuracy"],
                                "Precision":metrics["Precision"],
                                "Recall":metrics["Recall"],
                                "F1_Score":metrics["F1_score"]})
            if problem_type=="Regression":
                results.append({"Model":model_name,**metrics})


            



        results_df=pd.DataFrame(results)

        return results_df

            

