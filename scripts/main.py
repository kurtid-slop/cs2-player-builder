import pandas as pd

from common import PLAYERS_FILE

data = pd.read_csv(PLAYERS_FILE)


df = pd.DataFrame(data)
print(df)