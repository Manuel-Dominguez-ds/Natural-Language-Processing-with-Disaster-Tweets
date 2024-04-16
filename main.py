from Pipelines.Trainer import *

if __name__=='__main__':
    trainer = Trainer('Data/train.csv')
    trainer.orchestrator()
    scorer=Scorer('Data/test.csv')
    scorer.orchestrator()
