import pandas as pd
archivo = pd.read_csv("insurance.csv")
age_mean = archivo["age"].mean()
age_median = archivo["age"].median()
age_mode = archivo["age"].mode()

print("""
Media: %d
Moda: %d
Mediana:%d
""" % (age_mean, age_mode, age_median))
