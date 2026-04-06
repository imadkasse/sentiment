url = "https://raw.githubusercontent.com/laxmimerit/IMDB-Movie-Reviews-Large-Dataset-50k/master/IMDB-Dataset.csv"

import pandas as pd
df = pd.read_csv(url)

print(df.head())