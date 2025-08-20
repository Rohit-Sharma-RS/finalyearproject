import pandas as pd

big_file = r"D:\cmi-detect-behavior-with-sensor-data\test.csv"
small_file = r"dummy_test.csv"

df_small = pd.read_csv(big_file, nrows=2)

df_small.to_csv(small_file, index=False)

print(f"Saved 2 rows to {small_file}")
