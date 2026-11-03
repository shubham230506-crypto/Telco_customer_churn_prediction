Create table Telco_Customer(
	CustomerID Varchar(20),
	Count int ,
	Country Varchar(20),
	State Varchar(20),
	City Varchar(20),	
	Zip_Code int,	
	Lat_Long int,	
	Latitude int,	
	Longitude int,	
	Gender Varchar(20),
	Senior_Citizen Varchar(20),	
	Partner Varchar(20),
	Dependents Varchar(20),	
	Tenure_Months int,	
	Phone_Service Varchar(20),	
	Multiple_Lines Varchar(20),	
	Internet_Service Varchar(20),
	Online_Security Varchar(20),	
	Online_Backup Varchar(20),	
	Device_Protection Varchar(20),	
	Tech_Support Varchar(20),	
	Streaming_TV Varchar(20),	
	Streaming_Movies Varchar(20),	
	Contract Varchar(20),	
	Paperless_Billing Varchar(20),	
	Payment_Method Varchar(20),	
	Monthly_Charges int,
	Total_Charges int,
	Churn_Label Varchar(20),
	Churn_Value int,
	Churn_Score int,
	CLTV int,
	Churn_Reason  Varchar(150)

);
-- Change latitude and longitude to numeric (decimal) types
ALTER TABLE public.telco_customer ALTER COLUMN latitude TYPE NUMERIC(10, 6);
ALTER TABLE public.telco_customer ALTER COLUMN longitude TYPE NUMERIC(10, 6);
ALTER TABLE public.telco_customer ALTER COLUMN monthly_charges TYPE NUMERIC(10, 2);
ALTER TABLE public.telco_customer ALTER COLUMN total_charges TYPE NUMERIC(10, 2);
-- Ensure lat_long is text (if you haven't run this already)
ALTER TABLE public.telco_customer ALTER COLUMN lat_long TYPE VARCHAR(100);