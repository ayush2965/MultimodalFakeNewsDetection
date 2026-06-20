import chardet
import pandas as pd

with open("coda_text.csv", 'rb') as f:
    result = chardet.detect(f.read())
encoding = result['encoding']
data = pd.read_csv("IFND.csv", encoding=encoding)