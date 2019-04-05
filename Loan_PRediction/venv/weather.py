import pandas as pd

df = pd.read_csv("weather_p.csv")
# pd.options.display.max_columns = 9
# pd.options.display.max_rows = 4
print(df.head())
print(df.pivot(index="date", columns="city", values=""))
