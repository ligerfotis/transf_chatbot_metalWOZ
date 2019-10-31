import csv
import string
# Importing libraries
import pandas as pd

import csv

import numpy as np
filename = "subs/chatterbot_corpus/data.csv"
df = pd.read_csv(filename)

# Take a look at the first few rows
string.punctuation = '"#$%&\'()*+-/:<=>@[\\]^_`{|}~'
df.user = df.user.apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))
df.system = df.system.apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))

#string.punctuation = ["  ", "   ", "    ", "     ", "      "]
#df.user = df.user.apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))
#df.system = df.system.apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))

df.user = df.user.apply(lambda x: x.lower())
df.system = df.system.apply(lambda x: x.lower())

df['user'].str.strip()
df['system'].str.strip()

df.user = df.user.str.replace('  ', ' ')
df.system = df.system.str.replace('  ', ' ')

dest_filename = "subs/chatterbot_corpus/clean_data.csv"
df.to_csv(dest_filename, encoding='utf-8', index=False)
print("File:" + dest_filename + " created.")

"""
# Read csv file into a pandas dataframe
filename = "data/alldata.csv" + str(i) + ".csv"
#df = pd.read_csv(filename)
toAdd = ["user", "system"]
with open(filename, "r") as infile:
    reader = list(csv.reader(infile))
    reader.insert(0, toAdd)

with open(filename, "w") as outfile:
    writer = csv.writer(outfile)
    for line in reader:
        writer.writerow(line)
"""