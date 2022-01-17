from surprise import Reader, Dataset, SVD
import pandas as pd
import pickle

df = pd.read_csv("databelanja.csv")

data = Dataset.load_from_df(df, Reader())
trainset = data.build_full_trainset()

# #train
# model = SVD()
# model.fit(trainset)

# #dump
# with open("model.pkl", "wb") as f:
#     pickle.dump(model, f)
    
#load
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print(model.predict(1, "pepsodent"))