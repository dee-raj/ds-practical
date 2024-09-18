import json

def compare_json_files(file1, file2):
   # Read content of both JSON files
   with open(file1, 'r') as f1, open(file2, 'r') as f2:
      data1 = json.load(f1)
      data2 = json.load(f2)

   # Compare the contents of the JSON files
   if data1 == data2:
      print("The contents of the JSON files are the same.")
   else:
      print("The contents of the JSON files are different.")

# Provide the paths to the JSON files you want to compare
file1_path = 'a.json'
file2_path = 'b.json'

# Call the function to compare the JSON files
compare_json_files(file1_path, file2_path)
