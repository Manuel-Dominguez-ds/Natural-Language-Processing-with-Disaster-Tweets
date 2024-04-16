from Pipelines.Trainer import *

if __name__=='__main__':
    trainer = Trainer('Data/train.csv')
    trainer.orchestrator()
    # Crear scorer class y hacer predicciones para test data