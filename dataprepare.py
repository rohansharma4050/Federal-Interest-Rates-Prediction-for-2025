import pandas as pd

df = pd.read_excel('inflation data.xlsx')
df_ir = pd.read_csv('FEDFUNDS.csv')
df_ud = pd.read_csv('UNRATE.csv')
df_bd = pd.read_csv('AAA.csv')

result = (
    df.melt(id_vars=["Year"], var_name="Month", value_name="Inflation")
    .assign(
        DATE=lambda x: pd.to_datetime(
            x["Year"].astype(str) + "-" + x["Month"], format="%Y-%b"
        )
    )
    .sort_values("DATE")[["DATE", "Inflation"]]
)
result['DATE'] = result['DATE'].astype(str)
result.dropna(subset=["Inflation"],inplace=True)
result = pd.merge(result,df_ud, on='DATE', how='left')
result = pd.merge(result,df_bd, on='DATE', how='left')
result = pd.merge(result,df_ir, on='DATE', how='left')

result.rename(columns={'Inflation': 'Inflation Rate','AAA': 'Bonds Yield','UNRATE': 'Unemployment Rate', 'DATE' : 'Date'}, inplace = True)
result['Date'] = pd.to_datetime(result['Date'])
result.to_csv('finaldata.csv',index=False)
