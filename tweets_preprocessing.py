import pandas as pd

input_file = "filtered_tweets.csv"
output_file = "processed_tweets.csv"

df = pd.read_csv(input_file)
columns_to_drop = ["user_name", "user_location", "user_description", "user_created", "hashtags", "source"]
df = df.drop(columns=columns_to_drop)
df.to_csv(output_file, index=False)
