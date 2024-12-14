import pandas as pd
import re

bitcoin_df = pd.read_csv("bitcoin.csv", parse_dates=["Date"], date_format="%Y-%m-%d")

try:
    tweets_df_iter = pd.read_csv(
        "tweets.csv", iterator=True, chunksize=5000, on_bad_lines="error", encoding="utf-8", engine="python", low_memory=True
    )
    tweets_chunk_sample = next(tweets_df_iter)
    tweets_df_iter = pd.read_csv(
        "tweets.csv", iterator=True, chunksize=5000, on_bad_lines="error", encoding="utf-8", engine="python", low_memory=True
    )
    tweets_date_col = tweets_chunk_sample.columns[8]
    tweets_cols = tweets_chunk_sample.columns.tolist()
    bitcoin_cols = bitcoin_df.columns.tolist()

    with open("filtered_bitcoin.csv", "w", newline="", encoding="utf-8") as bitcoin_out, open("filtered_tweets.csv", "w", newline="", encoding="utf-8") as tweets_out:
        bitcoin_header = ",".join(bitcoin_cols)
        tweets_header = ",".join(tweets_cols)
        bitcoin_out.write(bitcoin_header + "\n")
        tweets_out.write(tweets_header + "\n")

        for tweets_chunk in tweets_df_iter:
            tweets_chunk["Date"] = pd.to_datetime(tweets_chunk[tweets_date_col].str.split().str[0], format="%Y-%m-%d", errors="coerce")
            merged_chunk = pd.merge(bitcoin_df, tweets_chunk, on="Date", how="inner")

            if not merged_chunk.empty:
                bitcoin_df_chunk = merged_chunk[bitcoin_cols]
                tweets_df_chunk = merged_chunk[tweets_cols]

                for col in tweets_df_chunk.columns:
                    if tweets_df_chunk[col].dtype == 'object':
                        tweets_df_chunk[col] = tweets_df_chunk[col].astype(str).apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))

                bitcoin_df_chunk.to_csv(bitcoin_out, header=False, index=False, encoding="utf-8")
                tweets_df_chunk.to_csv(tweets_out, header=False, index=False, encoding="utf-8")

except pd.errors.ParserError as e:
    print(f"Parser Error: {e}")