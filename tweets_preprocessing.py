import pandas as pd
import re

input_file = "filtered_tweets.csv"
output_file = "processed_tweets.csv"


def clean_text(text):
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df = pd.read_csv(input_file)
columns_to_drop = ["user_name", "user_location", "user_description", "user_created", "user_followers", "user_friends",
                   "user_favourites", "user_verified", "hashtags", "source", "is_retweet"]

df = df.drop(columns=columns_to_drop)
df["text"] = df["text"].apply(clean_text)

df.to_csv(output_file, index=False)
