from pathlib import Path
import pandas as pd
import sqlite3

conn=sqlite3.connect("Bank_Data_Model.db")
cursor=conn.cursor()
# cursor.execute("""CREATE TABLE Account(AccountID int, BranchCode int, AccountNumber int, CurrencyCode text, DateOpened Date ,Status int, DateClosed Date,AccountTypeCode text ,ProductID int, JoinedAccountIndicator text, AccountOfficer int, CurrentBalance float,CreatedDate date,CreatedBy int, ModifiedDate date, ModifiedBy
# int) """)
# cursor.execute("""CREATE TABLE AccountHolder(AccountHolderID int, AccountNumber int, CustomerID int, AuthorisedSignature text, CreatedDate Date ,CreatedBy int, ModifiedDate Date,ModifiedBy int) """)
# cursor.execute("""CREATE TABLE AccountStatus(AccountStatusID int, AccountStatus text, Description text, CreatedDate date, CreatedBy int ,ModifiedDate date, ModifiedBy int) """)
# cursor.execute("""CREATE TABLE AccountType(AccountTypeCode text, AccountType text, Description text, CreatedDate date, CreatedBy int ,ModifiedDate date, ModifiedBy int) """)
# cursor.execute("""CREATE TABLE Address(AddressID int, Address text, City text, County text, PostCode text ,CreatedDate date, CreatedBy int,ModifiedDate date,ModifiedBy int) """)
# cursor.execute("""CREATE TABLE AddressType(AddressTypeCode text, AddressType text, CreatedDate date, CreatedBy int , ModifiedDate date, ModifiedBy int) """)
# cursor.execute("""CREATE TABLE Branch(BranchCode int, BranchName text, City text, SortCode date , BranchManager int, AddressID int,BranchType text, CreatedDate date, CreatedBy int,ModifiedDate date, ModifiedBy int) """)
# cursor.execute("""CREATE TABLE Currency(CurrencyCode text, CurrencyName text, Country text,CreatedDate date, CreatedBy int,ModifiedDate date, ModifiedBy int) """)
# cursor.execute("""CREATE TABLE customer(CustomerID int, Title text, FirstName text,MiddleName text, LastName text,DateofBirth date,Gender text, RegistrationDate date,IdentificationDocumentCode text,ProofofAddressCode text,NumberofAccounts int,OccupationID int,MaritalStatusCode text,NumberofChildren int,MonthsinCurrentAddress int,EstimatedAnnualIncome float,HouseOwnershipCode text,CreatedDate date,CreatedBy int,ModifiedDate date, ModifiedBy int) """)
# cursor.execute("""CREATE TABLE CustomerAddress(CustomerAddressID int, CustomerID int, AddressID int,AddressTypeCode text, ValidFrom Date,ValidTo date,ActiveIndicator text,CreatedDate date,CreatedBy int,ModifiedDate date, ModifiedBy int) """)
# cursor.execute("""CREATE TABLE CustomerTelephoneNumber(CustomerTelephoneNumberID int, CustomerID int, TelephoneNumberID int,TelephoneNumberTypeCode text, ValidFrom Date,ValidTo Date,ActiveIndicator text,CreatedDate date,CreatedBy int,ModifiedDate date, ModifiedBy int) """)
# cursor.execute("""CREATE TABLE Employee(EmployeeID int, Login text, FirstName text,LastName text, Branch int,Title text,CreatedDate date,CreatedBy int,ModifiedDate date, ModifiedBy int) """)
# cursor.execute("""CREATE TABLE HouseOwnership(HouseOwnershipCode text, HouseOwnership text,CreatedDate date,CreatedBy int,ModifiedDate date, ModifiedBy int) """)
# cursor.execute("""CREATE TABLE IndentificationDocument(IdentificationDocumentCode text, IdentificationDocument text,CreatedDate date,CreatedBy int,ModifiedDate date, ModifiedBy int) """)
# cursor.execute("""CREATE TABLE InterestRate(SchemeID int, LowerLimit float,UpperLimit float,InterestRate float,CreatedDate date,CreatedBy int,ModifiedDate date, ModifiedBy int) """)
# cursor.execute("""CREATE TABLE MaritalStatus(MaritalStatusCode text, MaritalStatus text,CreatedDate date,CreatedBy int,ModifiedDate date, ModifiedBy int) """)
# cursor.execute("""CREATE TABLE Product(ProductID int, ProductName text,AccountTypeCode int, ProductType text,DateLaunched Date,ProductManager int,BalanceLimit float,CurrencyCode text,CreatedDate date,CreatedBy int,ModifiedDate date, ModifiedBy int) """)
# cursor.execute("""CREATE TABLE ProductInterestRate(ProductInterestRateID int, ProductID int,InterestRateSchemeID int,ValidFrom Date,ValidTo Date, ActiveFlag text,CreatedDate date,CreatedBy int,ModifiedDate date, ModifiedBy int) """)
# cursor.execute("""CREATE TABLE ProductType(ProductTypeCode text, ProductType text,CreatedDate date,CreatedBy int,ModifiedDate date, ModifiedBy int) """)
# cursor.execute("""CREATE TABLE ProofOfAddress(ProofOfAddressCode text, ProofOfAddress text,CreatedDate date,CreatedBy int,ModifiedDate date, ModifiedBy int) """)
# cursor.execute("""CREATE TABLE TelephoneNumber(TelephoneNumberID int, CountryCode int,AreaCode int,TelephoneNumber int,CreatedDate date,CreatedBy int,ModifiedDate date, ModifiedBy int) """)
# cursor.execute("""CREATE TABLE TelephoneNumberType(TelephoneNumberTypeCode text, TelephoneNumberType text,CreatedDate date,CreatedBy int,ModifiedDate date, ModifiedBy int) """)
# cursor.execute("""CREATE TABLE Trans(TransactionID int, TransactionDateTime text,UserID int,AccountID int , Amount float,AccountBalance float) """)


# df_Acc=pd.read_csv("C:\Project\PandasAI\Data_Model\Account.csv")
# df_Acc_Holder=pd.read_csv("C:\Project\PandasAI\Data_Model\AccountHolder.csv")
# df_Acc_Status=pd.read_csv("C:\Project\PandasAI\Data_Model\AccountStatus.csv")
# df_Acc_Type=pd.read_csv("C:\Project\PandasAI\Data_Model\AccountType.csv")
# df_addr=pd.read_csv("C:\Project\PandasAI\Data_Model\Address.csv")
# df_addtype=pd.read_csv("C:\Project\PandasAI\Data_Model\AddressType.csv")
# df_branch=pd.read_csv("C:\Project\PandasAI\Data_Model\Branch.csv")
# df_curr=pd.read_csv("C:\Project\PandasAI\Data_Model\Currency.csv")
# df_cust=pd.read_csv("C:\Project\PandasAI\Data_Model\customer.csv")
# df_cust_Add=pd.read_csv("C:\Project\PandasAI\Data_Model\CustomerAddress.csv")
# df_cut_tele=pd.read_csv("C:\Project\PandasAI\Data_Model\CustomerTelephoneNumber.csv")
# df_emp=pd.read_csv("C:\Project\PandasAI\Data_Model\Employee.csv")
# df_houseown=pd.read_csv("C:\Project\PandasAI\Data_Model\HouseOwnership.csv")
# df_ID=pd.read_csv("C:\Project\PandasAI\Data_Model\IndentificationDocument.csv")
# df_Acc_IR=pd.read_csv("C:\Project\PandasAI\Data_Model\InterestRate.csv")
# df_Acc_MS=pd.read_csv("C:\Project\PandasAI\Data_Model\MaritalStatus.csv")
# df_Acc_Prod=pd.read_csv("C:\Project\PandasAI\Data_Model\Product.csv")
# df_Acc_Prod_IR=pd.read_csv("C:\Project\PandasAI\Data_Model\ProductInterestRate.csv")
# df_Acc_ProdType=pd.read_csv("C:\Project\PandasAI\Data_Model\ProductType.csv")
# df_Acc_Proof=pd.read_csv("C:\Project\PandasAI\Data_Model\ProofOfAddress.csv")
# df_Acc_TeleNum=pd.read_csv("C:\Project\PandasAI\Data_Model\TelephoneNumber.csv")
# df_TelenumType=pd.read_csv("C:\Project\PandasAI\Data_Model\TelephoneNumberType.csv")
df_Trans=pd.read_csv("C:\Project\PandasAI\Data_Model\Transaction.csv")



# df_Acc.to_sql("Account",conn, if_exists='replace',index= False)
# df_Acc_Holder.to_sql("AccountHolder",conn, if_exists='replace',index= False)
# df_Acc_Status.to_sql("AccountStatus",conn, if_exists='replace',index= False)
# df_Acc_Type.to_sql("AccountType",conn, if_exists='replace',index= False)
# df_addr.to_sql("Address",conn, if_exists='replace',index= False)
# df_addtype.to_sql("AddressType",conn, if_exists='replace',index= False)
# df_branch.to_sql("Branch",conn, if_exists='replace',index= False)
# df_curr.to_sql("Currency",conn, if_exists='replace',index= False)
# df_cust.to_sql("customer",conn, if_exists='replace',index= False)
# df_cust_Add.to_sql("CustomerAddress",conn, if_exists='replace',index= False)
# df_cut_tele.to_sql("CustomerTelephoneNumber",conn, if_exists='replace',index= False)
# df_emp.to_sql("Employee",conn, if_exists='replace',index= False)
# df_houseown.to_sql("HouseOwnership",conn, if_exists='replace',index= False)
# df_ID.to_sql("IndentificationDocument",conn, if_exists='replace',index= False)
# df_Acc_IR.to_sql("InterestRate",conn, if_exists='replace',index= False)
# df_Acc_MS.to_sql("MaritalStatus",conn, if_exists='replace',index= False)
# df_Acc_Prod.to_sql("Product",conn, if_exists='replace',index= False)
# df_Acc_Prod_IR.to_sql("ProductInterestRate",conn, if_exists='replace',index= False)
# df_Acc_ProdType.to_sql("ProductType",conn, if_exists='replace',index= False)
# df_Acc_Proof.to_sql("ProofOfAddress",conn, if_exists='replace',index= False)
# df_Acc_TeleNum.to_sql("TelephoneNumber",conn, if_exists='replace',index= False)
# df_TelenumType.to_sql("TelephoneNumberType",conn, if_exists='replace',index= False)
df_Trans.to_sql("Trans",conn, if_exists='replace',index= False)

conn.commit()
conn.close()



