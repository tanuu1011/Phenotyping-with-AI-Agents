import pandas as pd

# load the three datasets
df1 = pd.read_csv("human_variant1_parsed.csv")
df2 = pd.read_csv("human_variant2_parsed.csv")
df3 = pd.read_csv("human_variant3_parsed.csv")

# add variant column
df1["variant"] = 1
df2["variant"] = 2
df3["variant"] = 3

# combine
df = pd.concat([df1, df2, df3], ignore_index=True)

# save combined dataset
df.to_csv("combined_igt_data_human.csv", index=False)

print("Combined dataset saved as combined_igt_data_human.csv")