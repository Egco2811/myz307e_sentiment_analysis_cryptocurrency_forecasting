import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style 
sns.set_theme(style="darkgrid")

# Read the CSV file
df = pd.read_csv('/content/bitcoin.csv')

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Filter data for the specified date range
mask = (df['Date'] >= '2021-02-09') & (df['Date'] <= '2022-07-11')
df_filtered = df.loc[mask]

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(df_filtered['Date'], 
         df_filtered['Close'], 
         label='Bitcoin Price', 
         color='#1f77b4', 
         linewidth=2)

# Customize the plot
plt.title('Bitcoin Price Movement', 
          fontsize=14, 
          pad=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# Rotate x-axis labels
plt.xticks(rotation=45)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('bitcoin_price_movement.png', 
            dpi=300, 
            bbox_inches='tight')

# Display the plot
plt.show()
