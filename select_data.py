import pandas as pd
from utils.logger import Logger

start_date = "2023_05_24"
num_try = "00"
data_path = f"./logs/{start_date}/{num_try}/process_time.csv"
df = pd.read_csv(data_path, sep=';')
log_path = "final/"
logger = Logger(path=log_path, filename="process_time.csv")

df = df[(df["seed"].isin([6694, 6695, 6696])) & (df["method"] != "monte carlo least core")]
for i in range(df.shape[0]):
    row_dict = df.iloc[i].to_dict()
    logger.log(row_dict)



