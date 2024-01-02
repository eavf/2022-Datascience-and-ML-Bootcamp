from lib_hd import *


# Načítať model
loaded_model = load_model(load_model_path)

# Načítať dáta
x_train, y_train, x_val, y_val, x_test, y_test, obr_ver, nr_ver = get_data()

print(f"Celková presnosť verifikovaných dát: {test_data_wo_btch(loaded_model, obr_ver, nr_ver) * 100:.2f}%")
print(f"Celková presnosť testovacích dát: {test_data_wo_btch(loaded_model, x_test, y_test) * 100:.2f}%")
print(f"Celková presnosť validačného setu dát: {test_data_wo_btch(loaded_model, x_val, y_val) * 100:.2f}%")
print(f"Celková presnosť trainingového setu dát: {test_data_wo_btch(loaded_model, x_train, y_train) * 100:.2f}%")
