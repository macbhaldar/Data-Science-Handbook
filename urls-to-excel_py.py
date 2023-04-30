#for writing to excel(xlsx) we will be needing XlsxWriter, please install it first if you don't have it!
try:
  import XlsxWriter
except ModuleNotFoundError:
  print("XlsxWriter is not installed!!")
  get_ipython().system("pip install XlsxWriter")

#to scrape a table from a webpage
from urllib.parse import urlparse,urlsplit
import requests
import pandas as pd
import os


urls=["https://en.wikipedia.org/wiki/List_of_current_heads_of_state_and_government",
      "http://www.inwea.org/wind-energy-in-india/wind-power-potential"
      ]
      
print(len(urls),"Urls Found")

#convert the sheetname- remove _ and - , put title case and remove spaces
def modify_name(my_str):
  replaced=my_str.replace("_", " ").replace("-", " ")
  return replaced.title().replace(" ","")


#get all tables from a url
def get_dataframes(url):
  html = requests.get(url).content
  df_list = pd.read_html(html)
  #print(len(df_list)," Dataframes Returned")
  return df_list

#if df is too small then don't add it
def filter_dfs(dfs_list,min_rows=10):
  new_dfs_list=[]
  for each_df in dfs_list:
    if(len(each_df)>min_rows):
      new_dfs_list.append(each_df)
  return new_dfs_list

#to avoid InvalidWorksheetName: Excel worksheet name 'StatesAndUnionTerritoriesOfIndia1' must be <= 31 chars.
def crop_name(name,thres=29):
  if len(name)<thres:
    return name
  else:
    return name[:thres]

#to get first n elements from list only
def crop_list(lst,thres=29):
  if len(lst)<thres:
    return lst
  else:
    return lst[:thres]

#converts urls to dataframes to excel sheets
#get_max= get the maximum number of tables from each url
#min_rows= the minimum number of rows in each table to save it to the excel sheet
#crop_name_thres= some excel sheets can get quite huge sheet names which blows up the code
#so crop the sheet name for the better purpose

def urls_to_excel(urls,excel_path=None,get_max=10,min_rows=0,crop_name_thres=29):
  excel_path=os.path.join(os.getcwd(),"Excel_Multiple_Sheets_Output.xlsx") if excel_path==None else excel_path
  writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
  i=0
  for url in urls:
    parsed=urlsplit(url)
    sheet_name=parsed.path.split('/')[-1]
    mod_sheet_name=crop_name(modify_name(sheet_name),thres=crop_name_thres)

    dfs_list=get_dataframes(url)
    filtered_dfs_list=filter_dfs(dfs_list,min_rows=min_rows)
    filtered_dfs_list=crop_list(filtered_dfs_list,thres=get_max)
    for each_df in filtered_dfs_list:
      print("Parsing Excel Sheet "," : ",str(i)+mod_sheet_name)
      i+=1
      each_df.to_excel(writer, sheet_name=str(i)+mod_sheet_name, index=True)
  writer.save()
  
urls_to_excel(urls,get_max=1,min_rows=10)
