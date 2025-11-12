import pandas as pd

df = pd.read_csv("dataset/dataset_small.csv")

no_rows = df[df["is_duplicate"] == "1"].head(20)
yes_rows = df[df["is_duplicate"] == "0"].head(10)
df = pd.concat([no_rows, yes_rows], ignore_index=True)
df = df[["question1", "question2"]]

new_df = pd.DataFrame({"text" : pd.concat([df["question1"], df["question2"]], ignore_index=True)})

new_df.to_csv("dataset/modified_dataset_question_2.csv", index=False)


