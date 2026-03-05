import pandas as pd

# Read the CSV file
df = pd.read_csv('war_stats.csv')

# Filter for players with $0 AAV
zero_aav_players = df[df['Player']== 'Carlos Correa']



# Display the results
print(zero_aav_players)