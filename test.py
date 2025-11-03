import pandas as pd

df = pd.read_csv("dataset/dataset_question.csv")

#cắt bớt dữ liệu để về 60 mb lấy phần đầu tiên
total_bytes = df.memory_usage(deep=True).sum()
if len(df) == 0:
    df = df
elif total_bytes <= 100_000_000:
    # already <= 100 MB, keep all
    pass
else:
    per_row = total_bytes / len(df)
    keep_rows = int(100_000_000 / per_row)
    keep_rows = max(1, keep_rows)
    df = df.iloc[:keep_rows]
df.to_csv("dataset/dataset_small.csv", index=False)
