print("------------Import pandas as pd.")
import pandas as pd

ecom = pd.read_csv("Ecommerce Purchases")

print("------------Check the head of the DataFrame.")
head = ecom.head(5)
print(head)
print("------------How many rows and columns are there?")
info = ecom.info()

print("------------What is the average Purchase Price?")
average_Purchase_Price = ecom['Purchase Price'].mean()
print(average_Purchase_Price)

print("------------What were the highest and lowest purchase prices?")
print(ecom['Purchase Price'].max())
print(ecom['Purchase Price'].min())

print("------------How many people have English 'en' as their Language of choice on the website?")
print(ecom[ecom['Language'] == 'en'].count())

print("------------How many people have the job title of Lawyer?")
print(ecom[ecom['Job'] == 'Lawyer'].info())

print(
    "------------How many people made the purchase during the AM and how many people made the purchase during PM ?")
print(ecom['AM or PM'].value_counts())

print("------------What are the 5 most common Job Titles?")
print(ecom['Job'].value_counts().head(5))

print(
    "------------Someone made a purchase that came from Lot: 90 WT , what was the Purchase Price for this transaction?")
print(ecom[ecom['Lot'] == '90 WT']['Purchase Price'])

print("------------What is the email of the person with the following Credit Card Number: 4926535242672853?")
print(ecom[ecom['Credit Card'] == 4926535242672853]['Email'])

print(
    "------------How many people have American Express as their Credit Card Provider *and* made a purchase above $95 ??")
print(ecom[(ecom['CC Provider'] == 'American Express') & (ecom['Purchase Price'] > 95)].count())

print("------------Hard: How many people have a credit card that expires in 2025??")
print(sum(ecom['CC Exp Date'].apply(lambda x: x[3:]) == '25'))

print("------------Hard: What are the top 5 most popular email providers/hosts (e.g. gmail.com, yahoo.com, etc...)??")
print(ecom['Email'].apply(lambda x: x.split('@')[1]).value_counts().head(5))
