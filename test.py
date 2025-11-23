import pandas as pd

df = pd.read_csv("dataset/data_semantic.csv")
df = df[["source_doc", "susp_doc"]]
df = df[:30]
df = pd.concat([df["source_doc"].astype(str), df["susp_doc"].astype(str)], ignore_index=True).to_frame(name="text")
df.to_csv("dataset/data_semantic_for_visual.csv", index=False)




