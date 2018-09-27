print("------------Import pandas as pd.------------")
import pandas as pd

print("------------Read Salaries.csv as a dataframe called sal.------------")
sal = pd.read_csv("Salaries.csv")

print("------------Check the head of the DataFrame.------------")
head = sal.head()

print("------------Use the .info() method to find out how many entries there are.------------")
info = sal.info()

basePay = sal['BasePay'].mean()
print(basePay)

OvertimePay = sal["OvertimePay"].max()
print(OvertimePay)

print(
    "What is the job title of  JOSEPH DRISCOLL ? Note: Use all caps, otherwise you may get an answer that doesn't match up (there is also a lowercase Joseph Driscoll).------------")
print(sal[sal["EmployeeName"] == "JOSEPH DRISCOLL"]['JobTitle'])

print("------------How much does JOSEPH DRISCOLL make (including benefits)?------------")
TotalPayBenefits = sal[sal["EmployeeName"] == "JOSEPH DRISCOLL"]['TotalPayBenefits']
print("------------TotalPayBenefits is: " + TotalPayBenefits.__str__())

print("------------What is the name of highest paid person (including benefits)?------------")
benefits = sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()]["EmployeeName"]
print(benefits)

print(
    "What is the name of lowest paid person (including benefits)? Do you notice something strange about how much he or she is paid?------------")
lowest_paid_person = sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].min()]["EmployeeName"]
print(lowest_paid_person)

print("------------What was the average (mean) BasePay of all employees per year? (2011-2014) ?------------")
average = sal.groupby('Year').mean()['BasePay']
print(average)

print("------------How many unique job titles are there?------------")
print(sal['JobTitle'].nunique())

print("------------What are the top 5 most common jobs?------------")
print(sal['JobTitle'].value_counts().head(5))

print(
    "How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurence in 2013?)------------")
print(sum(sal[sal['Year'] == 2013]['JobTitle'].value_counts() == 1))

print("------------How many people have the word Chief in their job title? (This is pretty tricky)------------")


def chief_string(title):
    if 'chief' in title.lower():
        return True
    else:
        return False


print(sum(sal['JobTitle'].apply(lambda x: chief_string(x))))

print("------------Bonus: Is there a correlation between length of the Job Title string and Salary?------------")
sal['title_len'] = sal['JobTitle'].apply(len)
print(sal[['title_len', 'TotalPayBenefits']].corr())


## Output
------------Import pandas as pd.------------
------------Read Salaries.csv as a dataframe called sal.------------
------------Check the head of the DataFrame.------------
------------Use the .info() method to find out how many entries there are.------------
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 148654 entries, 0 to 148653
Data columns (total 13 columns):
Id                  148654 non-null int64
EmployeeName        148654 non-null object
JobTitle            148654 non-null object
BasePay             148045 non-null float64
OvertimePay         148650 non-null float64
OtherPay            148650 non-null float64
Benefits            112491 non-null float64
TotalPay            148654 non-null float64
TotalPayBenefits    148654 non-null float64
Year                148654 non-null int64
Notes               0 non-null float64
Agency              148654 non-null object
Status              0 non-null float64
dtypes: float64(8), int64(2), object(3)
memory usage: 14.7+ MB
66325.4488404877
245131.88
What is the job title of  JOSEPH DRISCOLL ? Note: Use all caps, otherwise you may get an answer that doesn't match up (there is also a lowercase Joseph Driscoll).------------
24    CAPTAIN, FIRE SUPPRESSION
Name: JobTitle, dtype: object
------------How much does JOSEPH DRISCOLL make (including benefits)?------------
------------TotalPayBenefits is: 24    270324.91
Name: TotalPayBenefits, dtype: float64
------------What is the name of highest paid person (including benefits)?------------
0    NATHANIEL FORD
Name: EmployeeName, dtype: object
What is the name of lowest paid person (including benefits)? Do you notice something strange about how much he or she is paid?------------
148653    Joe Lopez
Name: EmployeeName, dtype: object
------------What was the average (mean) BasePay of all employees per year? (2011-2014) ?------------
Year
2011    63595.956517
2012    65436.406857
2013    69630.030216
2014    66564.421924
Name: BasePay, dtype: float64
------------How many unique job titles are there?------------
2159
------------What are the top 5 most common jobs?------------
Transit Operator                7036
Special Nurse                   4389
Registered Nurse                3736
Public Svc Aide-Public Works    2518
Police Officer 3                2421
Name: JobTitle, dtype: int64
How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurence in 2013?)------------
202
------------How many people have the word Chief in their job title? (This is pretty tricky)------------
627
------------Bonus: Is there a correlation between length of the Job Title string and Salary?------------
                  title_len  TotalPayBenefits
title_len          1.000000         -0.036878
TotalPayBenefits  -0.036878          1.000000
