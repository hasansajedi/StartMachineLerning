import pandas as pd


def add98(x):
    x = str(x)
    if x.startswith('0'):
        return '98' + x[1:-1]
    else:
        return '98' + x[0:-1]


i = 0
for chunk in pd.read_csv('data/tel970913.csv', encoding="iso8859_16", header=None, names=['tel'], chunksize=100000,
                         lineterminator='\n'):
    if i == 0:
        chunk = chunk.iloc[1:]
    i += 1
    chunk['tel1'] = chunk['tel'].apply(add98)
    x = chunk.groupby('tel1').size()
    # print(x)
    chunk.to_csv("data/exp.csv", columns=['tel1'], mode='a', header=False, index=False)
