import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,f1_score,precision_score,recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from datetime import date
import pickle
import traceback
from Functions import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec, Doc2Vec
#import tensorflow as tf


'''
Build a training pipeline (training.py) where you will read this table, 
prepare it for modelling, run a hyperparameters search, store the best
trained model in any desired format, and store a text or pdf file with 
some evaluation metrics to describe the model.
'''


class Trainer():

    def __init__(self,df_path) -> None:
        print("\n > Initialized Training Pipeline.")
        self.df_path = df_path

    def orchestrator(self):
        try:
            print("\n > Starting Training process.")
            
            dataset = self.read_data(path=self.df_path)
            df_encoded = self.preprocessing(dataset)
            X_train, X_val, y_train, y_val = self.train_validation_split(df_encoded)
            # Escribir un for loop para probar diferentes vectorizadores automaticamente?
            X_train, X_val, chosen_vectorizer = self.vectorizer(X_train = X_train, X_val = X_val)
            
            self.X_train = X_train
            self.X_val = X_val
            self.y_train = y_train
            self.y_val = y_val

            self.train_models()
            self.evaluate_and_save_results(chosen_vectorizer)
            print("\n > Finished Training process succesfully")
        except Exception as e:
            print(f"\n > Exception during the training process: {e}")
            traceback.print_exc()
    
    def read_data(self,path):
        print("\n > Reading data.")
        df=pd.read_csv(path)
        return df
    
    def preprocessing(self, df):
        print("\n > Preprocessing text data.")
        df.text=df.text.apply(lambda x:preprocess(x))
        print("\n > Text data preprocessed.")
        return df

    def train_validation_split(self, df):
        y = df.target
        X = df.text
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        splits = X_train, X_val, y_train, y_val
        print(f"\n > Performed an 70/30 split. Training set has {X_train.shape[0]} examples. Validation set has {X_val.shape[0]} examples.")
        return splits
    
    def vectorizer(self,X_train,X_val):
        X_train=[text for text in X_train]
        X_val=[text for text in X_val]
        print('\nAvailable Vectorizers: \n-CountVectorizer \n-TfidfVectorizer ')
        
        vectorizer=input("\nChoose Vectorizer: ")
                
        # Validar la elección del usuario
        while vectorizer not in ['CountVectorizer', 'TfidfVectorizer']:
            print("Vectorizer not available. Please try again.")
            vectorizer = input("Choose Vectorizer: ")
            
        if vectorizer=='CountVectorizer':
            vectorizer = CountVectorizer()
            X_train_embedding = vectorizer.fit_transform(X_train) 
            X_val_embedding = vectorizer.transform(X_val)  
            with open('vectorizer.pkl', 'wb') as file:
                print(f"\n > Exporting vectorizer to pkl file 'best_model.pkl'⬇️💾")
                pickle.dump(vectorizer, file)
            return X_train_embedding,X_val_embedding,'CountVectorizer'
        
        
        elif vectorizer=='TfidfVectorizer':
            vectorizer = TfidfVectorizer()
            X_train_embedding = vectorizer.fit_transform(X_train) 
            X_val_embedding = vectorizer.transform(X_val)  
            with open('vectorizer.pkl', 'wb') as file:
                print(f"\n > Exporting vectorizer to pkl file 'best_model.pkl'⬇️💾")
                pickle.dump(vectorizer, file)
            return X_train_embedding,X_val_embedding,'TfidfVectorizer'
    
    def train_models(self):
        print("\n > Training models. Performing HyperParameter tunning with Gridsearch cross validation. This could take a while..🧠")
        # TODO merge al metrics in a single dictionary
        self.metrics={
            'Mean Squared Error':0,
            'Mean Absolut Error':0,
            'Root Mean Squared Deviation':0,
            'R2':0
        }
        self.precision = []
        self.recall = []
        self.accuracy = []
        self.f1 = []
        self.model_instances = {}
        self.models = {'Models':[],'Accuracy':[],'Precision':[],'Recall':[],'F1':[]}

        self.train_logistic_regression()
        self.train_random_forest_classifier()
        self.train_neural_network_classifier()
  

    def train_logistic_regression(self):
        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val

        model = LogisticRegression(random_state=42)
        linear_grid_search = GridSearchCV(
            model,
            param_grid={},
            cv=5,
            n_jobs=-1
        )
        linear_grid_search.fit(X_train, y_train)

        best_linear = linear_grid_search.best_estimator_

        y_pred_train = best_linear.predict(X_train)
        y_pred = best_linear.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        precision=precision_score(y_val,y_pred)
        recall=recall_score(y_val,y_pred)
        self.accuracy.append(accuracy)
        self.f1.append(f1)
        self.precision.append(precision)
        self.recall.append(recall)
        self.models['Models'].append('Logistic Regression')
        self.models['Accuracy'].append(accuracy)
        self.models['Precision'].append(precision)
        self.models['Recall'].append(recall)
        self.models['F1'].append(f1)
        self.model_instances.setdefault('Logistic Regression',best_linear)
  
        print("\n > Results for Logistic Regression\n")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {round(precision)}")
        print(f"F1 Score: {round(f1,2)}")
        print(f"Recall: {round(recall,2)}")


    def train_random_forest_classifier(self):
        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        linear_grid_search = GridSearchCV(
            model,
            param_grid={},
            cv=5,
            n_jobs=-1
        )
        linear_grid_search.fit(X_train, y_train)

        best_linear = linear_grid_search.best_estimator_

        y_pred_train = best_linear.predict(X_train)
        y_pred = best_linear.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        precision=precision_score(y_val,y_pred)
        recall=recall_score(y_val,y_pred)
        self.accuracy.append(accuracy)
        self.f1.append(f1)
        self.precision.append(precision)
        self.recall.append(recall)
        self.models['Models'].append('Random Forest Classifier')
        self.models['Accuracy'].append(accuracy)
        self.models['Precision'].append(precision)
        self.models['Recall'].append(recall)
        self.models['F1'].append(f1)
        self.model_instances.setdefault('Random Forest Classifier',best_linear)
  
        print("\n > Results for Random Forest Classifier\n")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {round(precision)}")
        print(f"F1 Score: {round(f1,2)}")
        print(f"Recall: {round(recall,2)}")
    
    def train_neural_network_classifier(self):
        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val

        model = tf.keras.Sequential([
        tf.keras.layers.Dense(8,activation='relu',input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1, activation='sigmoid')])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_val, y_val))

        y_pred_train = model.predict(X_train)
        y_pred = model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        precision=precision_score(y_val,y_pred)
        recall=recall_score(y_val,y_pred)
        self.accuracy.append(accuracy)
        self.f1.append(f1)
        self.precision.append(precision)
        self.recall.append(recall)
        self.models['Models'].append('Neural Network Classifier')
        self.models['Accuracy'].append(accuracy)
        self.models['Precision'].append(precision)
        self.models['Recall'].append(recall)
        self.models['F1'].append(f1)
        self.model_instances.setdefault('Neural Network Classifier',model)
  
        print("\n > Results for Neural Network Classifier\n")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {round(precision)}")
        print(f"F1 Score: {round(f1,2)}")
        print(f"Recall: {round(recall,2)}")


    def evaluate_and_save_results(self,vectorizer):
        print(f"\n > Evaluating and saving best model with vectorizer {vectorizer}.")
        models_results=pd.DataFrame(self.models)
        # save results to txt
        filename = f'training_output - {date.today()} - {vectorizer}.csv'
        with open(filename, 'a') as file:
            file.write("Models, Accuracy, Precision, Recall, F1\n")
            for i in range(models_results.shape[0]):
                line = str(models_results.Models[i]) + "," + str(models_results.Accuracy[i])+","+ str(models_results.Precision[i])+ "," + str(models_results.Recall[i]) + "," + str(models_results.F1[i])
                file.write(line + "\n")
        print(f"Training results have been saved to {filename}")

        best_model = self.get_best_model()
        self.save_model(best_model)

    def get_best_model(self):
        # get best model based on the RMSE Metric
        models_results=pd.DataFrame(self.models)
        best_model_name=models_results.loc[models_results['Accuracy'].idxmax()]['Models']
        print(f"\n ----> 🧠 Best model is {best_model_name} 🧠 <----")
        return self.model_instances[best_model_name]

    def save_model(self, best):
        # Dump model as pkl file
        with open('best_model.pkl', 'wb') as file:
            print(f"\n > Exporting best model to pkl file 'best_model.pkl'⬇️💾")
            pickle.dump(best, file)
            

            

    
