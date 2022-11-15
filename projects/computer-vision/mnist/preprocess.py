import pandas as pd

class Preprocess:
    def ReturnLabels(self, df):
        return df['label']
    def DatasetToNumpy(self, df):
        df = df.drop(columns = ['label'])
        return df.to_numpy()
    def GetLabelDummies(self, labels):
        return pd.get_dummies(labels).to_numpy()