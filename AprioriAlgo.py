import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

# Path to the dataset
path = "./store_data.csv"
# Load the dataset
dataset = pd.read_csv(path)
dataset.head()

# Prepare the records for the apriori algorithm
records = []
for i in range(0, 7500):
    transaction = []
    data = dataset.iloc[i].dropna()
    for item in data:
        transaction.append(str(item))
    records.append(transaction)

# Apply the apriori algorithm
association_rules = apriori(
    records, min_support=0.005, min_confidence=0.2, min_lift=3, min_length=2
)
association_results = list(association_rules)

# Print the association rules
for item in association_results:
    print(list(item[2][0][0]), '->', list(item[2][0][1]))