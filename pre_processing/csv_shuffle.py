import pandas as pd
import numpy as np

# Load the data from CSV files
x_train = pd.read_csv('../datas/SEDS/old/x_train.csv').values
y_train = pd.read_csv('../datas/SEDS/old/y_train.csv').values
x_val = pd.read_csv('../datas/SEDS/old/x_val.csv').values
y_val = pd.read_csv('../datas/SEDS/old/y_val.csv').values
x_test = pd.read_csv('../datas/SEDS/old/x_test.csv').values
y_test = pd.read_csv('../datas/SEDS/old/y_test.csv').values

# Combine features and labels
train_data = np.column_stack((x_train, y_train))
val_data = np.column_stack((x_val, y_val))
test_data = np.column_stack((x_test, y_test))

# Shuffle the data
np.random.shuffle(train_data)
np.random.shuffle(val_data)
np.random.shuffle(test_data)

# Split the features and labels back apart
x_train_shuffled = train_data[:, :-1]
y_train_shuffled = train_data[:, -1]
x_val_shuffled = val_data[:, :-1]
y_val_shuffled = val_data[:, -1]
x_test_shuffled = test_data[:, :-1]
y_test_shuffled = test_data[:, -1]

# Optionally, save the shuffled data back to new CSV files
pd.DataFrame(x_train_shuffled).to_csv('../datas/SEDS/x_train.csv', index=False)
pd.DataFrame(y_train_shuffled).to_csv('../datas/SEDS/y_train.csv', index=False)
pd.DataFrame(x_val_shuffled).to_csv('../datas/SEDS/x_val.csv', index=False)
pd.DataFrame(y_val_shuffled).to_csv('../datas/SEDS/y_val.csv', index=False)
pd.DataFrame(x_test_shuffled).to_csv('../datas/SEDS/x_test.csv', index=False)
pd.DataFrame(y_test_shuffled).to_csv('../datas/SEDS/y_test.csv', index=False)