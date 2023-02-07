import pandas as pd

a = pd.DataFrame([1, 2, 3])
b = pd.DataFrame([1, 2, 3])
print(a.apply(lambda x:x))