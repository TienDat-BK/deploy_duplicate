import pandas as pd

df = pd.read_csv("dataset/dataset_small.csv")

df = df[["question1", "question2"]]

df["question1"] = df["question1"] + df["question2"]
df = df.drop(columns=["question2"])

df.to_csv("dataset/modified_dataset_question.csv", index=False)

