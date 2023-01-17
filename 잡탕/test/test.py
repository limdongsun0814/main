from xlrd import open_workbook

wb = open_workbook('data.xlsx')
sheet = wb.sheet_by_index(0)

for j in range(1,361):
  data_file_name='theta '+str(j)+'.csv'
  f_write = open(data_file_name, 'w')
  wr = csv.writer(f_write)
  print(data_file_name)
  for i in range(100):
    # print(sheet.cell_value(i, 0))
    data ='a'+str(i)
    wr.writerow([data,sheet.cell_value(i, j-1)])