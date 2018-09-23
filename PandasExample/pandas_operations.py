import pandas as pd

df = pd.DataFrame({
    'col1': [1, 2, 3, 4],
    'col2': [444, 555, 666, 444],
    'col3': ['abc', 'def', 'ghi', 'xyz']})

print("-------------------Start basic functions-------------------")
print(df.head())  # Get head of data

print(df['col2'].unique())  # print unique values in column with array
print(df['col2'].nunique())  # print number of unique values in column
print(df['col2'].value_counts())  # print number of times repeated values in column

print(df[df['col1'] > 2])
print(df[(df['col1'] > 2) & (df['col2'] == 444)])

print(df.columns)
print(df.index)
print(df.sort_values(by='col2'))
print(df.isnull())

print("-------------------End basic functions-------------------")

print("-------------------Start apply-------------------")
def times2(x):
    return x * 2

print(df['col1'].apply(times2))  # all of items in col1 * 2
print(df['col3'].apply(len))
print(df['col3'].apply(len))
print(df['col2'].apply(lambda x: x * 2))
print("-------------------End pivot-------------------")

print("-------------------Start pivot-------------------")
data = {'A':['foo','foo','foo','bar','bar','bar'],
     'B':['one','one','two','two','one','one'],
       'C':['x','y','x','y','x','y'],
       'D':[1,3,2,5,4,1]}

df = pd.DataFrame(data)
print(df.pivot_table(values='D',index=['A', 'B'],columns=['C']))
print("-------------------End pivot-------------------")


## Output
>-------------------Start basic functions-------------------
   col1  col2 col3
0     1   444  abc
1     2   555  def
2     3   666  ghi
3     4   444  xyz
[444 555 666]
3
444    2
555    1
666    1
Name: col2, dtype: int64
   col1  col2 col3
2     3   666  ghi
3     4   444  xyz
   col1  col2 col3
3     4   444  xyz
Index(['col1', 'col2', 'col3'], dtype='object')
RangeIndex(start=0, stop=4, step=1)
   col1  col2 col3
0     1   444  abc
3     4   444  xyz
1     2   555  def
2     3   666  ghi
    col1   col2   col3
0  False  False  False
1  False  False  False
2  False  False  False
3  False  False  False
-------------------End basic functions-------------------
-------------------Start apply-------------------
0    2
1    4
2    6
3    8
Name: col1, dtype: int64
0    3
1    3
2    3
3    3
Name: col3, dtype: int64
0    3
1    3
2    3
3    3
Name: col3, dtype: int64
0     888
1    1110
2    1332
3     888
Name: col2, dtype: int64
-------------------End pivot-------------------
-------------------Start pivot-------------------
C          x    y
A   B            
bar one  4.0  1.0
    two  NaN  5.0
foo one  1.0  3.0
    two  2.0  NaN
-------------------End pivot-------------------
