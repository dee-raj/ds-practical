_parsed_rows = []

def parse_csv():
   import csv
   _file_path = "Feature Selection/deaths-by-sex-and-age.csv"
   _index = {
      "Period":0,
      "Sex":1,
      "Age":2,
      "Count":3
   }

   global _parsed_rows
   with open(_file_path, 'r') as f:
      reader = csv.reader(f)
      next(reader, None)
      for row in reader:
         _parsed_rows.append({
            "Period": row[_index["Period"]],
            "Sex": row[_index["Sex"]],
            "Age": row[_index["Age"]],
            "Count": row[_index["Count"]]
         })

def get_count_for_gender(gender_name):
   count = 0
   for row in _parsed_rows:
      if row["Sex"] == gender_name:
         count += int(row['Count'])
   return count


if __name__== "__main__":
   parse_csv()
   tot_count = get_count_for_gender("Total")
   print(tot_count)
   