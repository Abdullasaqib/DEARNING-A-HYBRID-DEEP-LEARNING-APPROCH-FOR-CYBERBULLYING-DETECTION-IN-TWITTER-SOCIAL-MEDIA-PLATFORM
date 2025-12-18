import pandas as pd

try:
    df = pd.read_csv('cyberbullying_tweets.csv')
    print("Columns:", df.columns.tolist())
    print("\nUnique labels in 'cyberbullying_type':")
    print(df['cyberbullying_type'].value_counts())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nSample tweets per class:")
    print(df.groupby('cyberbullying_type').first())
except Exception as e:
    print(f"Error: {e}")
