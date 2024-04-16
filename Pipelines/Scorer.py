import pickle
import numpy as np
import pandas as pd
from Functions import *
from datetime import date

class Scorer():
    def __init__(self, df_path) -> None:
        print("\n > Initialized Scoring Pipeline.")
        self.df_path = df_path

    def orchestrator(self):
        print("\n > Initialized Scoring process.")
        test_data=pd.read_csv(self.df_path)
        print("\n > Test data read.")
        sample_data=pd.DataFrame(test_data.text)
        print("\n > Loading model.")
        model = self.load_model()
        print("\n > Model loaded.")
        vectorizer=self.load_vectorizer()
        variables = test_data.text
        print("\n > Preprocessing and Vectorizing.")
        encoded = self.preprocessing(variables)
        print("\n > Loading vectorizer.")
        vectorized = self.vectorizer(encoded,vectorizer)
        print("\n > Making predictions.")
        predictions = self.predict(model=model, X=vectorized)
        sample_data["predictions"] = predictions
        print('\n------------------\nPredictions for Test Data\n------------------\n')
        print(sample_data)
        filename= f'Predictions - {date.today()}.csv'
        with open(filename, 'a') as file:
            file.write("Text,Predictions\n")
            for i in range(sample_data.shape[0]):
                line = str(sample_data.text[i]) + "," + str(sample_data.predictions[i])
                file.write(line + "\n")
        print(f"Training results have been saved to {filename}")

    def load_model(self):
        print("\n > Loading BestModel from memory.")
        with open('best_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
            return loaded_model
        
    def load_vectorizer(self):
        print("\n > Loading Vectorizer from memory.")
        with open('vectorizer.pkl', 'rb') as file:
            vectorizer = pickle.load(file)
            return vectorizer
        
    def vectorizer(self,X_test,vectorizer):
        X_test=[text for text in X_test]
        X_test_embedding=vectorizer.transform(X_test) 
        return X_test_embedding
        
    def preprocessing(self, variables):
        print("\n > Preprocessing text data.")
        variables=variables.apply(lambda x:preprocess(x))
        print("\n > Text data preprocessed.")
        return variables
    
    def predict(self, model, X):
        print("\n > Predicting charges...")
        predictions = model.predict(X)
        return predictions


