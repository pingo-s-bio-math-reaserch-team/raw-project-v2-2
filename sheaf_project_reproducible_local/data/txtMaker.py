from openpyxl import Workbook
from openpyxl import load_workbook

filePath = "twentyFourAndUp.xlsx"
wb = load_workbook(filePath)
sheet = wb.active

size = 422
headers = ["B", "C", "E", "F", "G", "H", "I", "J", "K", "L", "M", "O", "P", "Q", "R", "W", "X", "Y", "Z", "AA", "AB", "AC", "AF", "AG", "AH", "AI", "AJ", "AK", "AL", "AQ", "AU", "AV", "AW", "AX", "AY", "AZ", "BA", "BC", "BD", "BH"]
f = open("data.txt", "w")

for j in range(size-1):
    data = ""
    for i in headers:
        data += str(sheet[i+str(j+1)].value).replace(" ", "") + " "
    data += "\n"
    f.write(data)

f.close()
    
