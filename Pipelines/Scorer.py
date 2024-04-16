import pickle
import numpy as np
import pandas as pd
from Functions import *

class Scorer():
    def __init__(self, df_path) -> None:
        print("\n > Initialized Scoring Pipeline.")
        self.df_path = df_path

    def orchestrator(self):
        print("\n > Initialized Scoring process.")
        test_data=pd.read_csv(self.df_path)
        sample_data=pd.DataFrame(test_data.text)
        model = self.load_model()
        variables = test_data.drop("target", axis = 1)
        encoded = self.preprocessing(variables)
        vectorized = self.vectorizer(encoded)
        predictions = self.predict(model=model, X=vectorized)
        sample_data["predicted_charges"] = predictions
        print('\n------------------\nPredictions for Test Data\n------------------\n')
        print(sample_data)

    def load_model(self):
        print("\n > Loading BestModel from memory.")
        with open('best_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
            return loaded_model
        
    def vectorizer(self,X_test):
        X_test=[text for text in X_test]
        print('\nAvailable Vectorizers: \n-CountVectorizer \n-TfidfVectorizer ')
        
        vectorizer=input("\nChoose Vectorizer: ")
                
        # Validar la elección del usuario
        while vectorizer not in ['CountVectorizer', 'TfidfVectorizer']:
            print("Vectorizer not available. Please try again.")
            vectorizer = input("Choose Vectorizer: ")
            
        if vectorizer=='CountVectorizer':
            vectorizer = CountVectorizer()
            X_test = vectorizer.fit_transform(X_train) 
            return X_test,'CountVectorizer'
        
        elif vectorizer=='TfidfVectorizer':
            vectorizer = TfidfVectorizer()
            X_test = vectorizer.fit_transform(X_train) 
            returnX_test,'TfidfVectorizer'
        
    def preprocessing(self, df):
        print("\n > Preprocessing text data.")
        df.text=df.text.apply(lambda x:preprocess(x))
        print("\n > Text data preprocessed.")
        return df
    
    def predict(self, model, X):
        print("\n > Predicting charges...")
        predictions = model.predict(X)
        return predictions
