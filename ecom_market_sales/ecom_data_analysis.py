import pandas as pd


def read_csv(filepath):
    return pd.read_csv(filepath)

def read_xlsx(filepath):
    return pd.read_excel(filepath)


def question_2():
  online_sales_df = read_csv("data/Online_Sales.csv")
  online_sales_df["Month"] = pd.to_datetime(online_sales_df["Transaction_Date"]).dt.strftime('%b')
  grup = online_sales_df.groupby(["CustomerID", "Month"], )["Transaction_Date"].min()
  grup = grup.reset_index()
  customer_month_acquisition = grup.groupby("Month")["CustomerID"].count()
  print(f"Total acquistion month wise :  \n {customer_month_acquisition}")

def question_1():
  """
  Identify the months with the highest and lowest acquisition count.
  What strategies could be implemented to address the fluctuations and ensure consistent growth throughout the year
  :return:
  """
  # customer_data_df = read_xlsx("data/CustomersData.xlsx")
  # print("Customer Data Info: \n", customer_data_df.info())
  # print("Customer Dataset head : \n", customer_data_df.head())
  # print("--" * 50)
  # print("\n\n")

  # discount_df = read_csv("data/Discount_Coupon.csv")
  # print("discount Data Info: \n", discount_df.info())
  # print(discount_df[discount_df["Month"]== "Feb"].groupby("Discount_pct")["Product_Category"].count())
  # print(discount_df[discount_df["Month"]== "Aug"].groupby("Discount_pct")["Product_Category"].count())
  # print(discount_df[(discount_df["Product_Category"] == "Apparel") & (discount_df["Month"] == "Aug")])
  # print(discount_df[(discount_df["Product_Category"] == "Apparel") & discount_df["Month"] == "Feb"])


  market_spend_df = read_csv("data/Marketing_Spend.csv")
  market_spend_df["Date"] = pd.to_datetime(market_spend_df["Date"]).dt.strftime('%b')
  market_spend_df = market_spend_df.rename(columns={'Date': 'Month'})
  market_spend_offline_aug_feb  = market_spend_df[market_spend_df["Month"].isin(["Aug", "Feb"])].groupby("Month")["Offline_Spend"].sum()
  market_spend_online_aug_feb= market_spend_df[market_spend_df["Month"].isin(["Aug", "Feb"])].groupby("Month")["Online_Spend"].sum()
  print(f"Difference in the market spend offline :\n  {market_spend_offline_aug_feb} \n Also in the online: {market_spend_online_aug_feb}")


  #
  # tax_amount_df = read_xlsx("data/Tax_amount.xlsx")
  # print("Tax Amount Info: \n", tax_amount_df.info())
  # print("Tax Amount  head : \n", tax_amount_df.head())
  # print("--" * 50)
  # print("\n\n")


  online_sales_df = read_csv("data/Online_Sales.csv")
  online_sales_df["Month"] = pd.to_datetime(online_sales_df["Transaction_Date"]).dt.strftime('%b')
  grup = online_sales_df.groupby(["CustomerID", "Month"], )["Transaction_Date"].min()
  grup= grup.reset_index()
  customer_month_acquisition= grup.groupby("Month")["CustomerID"].count()
  print(f"Total aquistion month wise :  \n {customer_month_acquisition}")
  max_count = customer_month_acquisition.max()
  min_count = customer_month_acquisition.min()

  max_month = customer_month_acquisition.idxmax()
  min_month = customer_month_acquisition.idxmin()
  print(f" {max_count} acquistion happen for month : {max_month} ")
  print(f"{min_count} acquistion happen for month : {min_month}")
  result = online_sales_df[
    (online_sales_df["Coupon_Status"] == "Used") &
    (online_sales_df["Month"].isin(["Aug", "Feb"]))
    ].groupby(["Month", "Product_Category"])["Coupon_Status"].count()

  print(f"Coupon Used Difference in the month of Aug and Feb \n: {result}")

  # print(f"Coupon Used Difference in the month of Aug and Feb \n: {online_sales_df[(online_sales_df["Coupon_Status"] == "Used") & (online_sales_df["Month"].isin(["Aug", "Feb"]))].groupby(["Month", "Product_Category"])["Coupon_Status"].count()}")
  delivery_charges_diff = online_sales_df[
    (online_sales_df["Product_Category"] == "Apparel") &
    (online_sales_df["Month"].isin(["Aug", "Feb"]))
    ][["Delivery_Charges", "Product_Category", "Quantity"]] \
    .sort_values(by="Delivery_Charges", ascending=False)
  print(f"Difference of delivery charges in month of Aug n Feb \n {delivery_charges_diff}")


  # online_sales_customer_df = pd.merge(online_sales_df,customer_data_df, on="CustomerID", how="left" )
  #
  # online_sales_customer_discount_df = pd.merge(online_sales_customer_df,discount_df, on=["Product_Category", "Month"], how="left" )
  #
  # online_sales_customer_discount_tax_df = pd.merge(online_sales_customer_discount_df, tax_amount_df, on="Product_Category", how="left")
  # online_sales_customer_discount_tax_df["revenue"] = online_sales_customer_discount_tax_df["Quantity"]*online_sales_customer_discount_tax_df["Avg_Price"] * (1-online_sales_customer_discount_tax_df["Discount_pct"]/100) * (1 + online_sales_customer_discount_tax_df["GST"]) + online_sales_customer_discount_tax_df["Delivery_Charges"]
  # print(online_sales_customer_discount_tax_df.info())
  #
  # customer_acusition_months = online_sales_customer_discount_tax_df.groupby('Month')
  # customer_acusition_months = customer_acusition_months.reset_index()
  # print(customer_acusition_months)


if __name__=='__main__':
    question_1()