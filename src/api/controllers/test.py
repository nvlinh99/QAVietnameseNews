import pickle

file_path = '/Users/linhnv/Coding/HCMUS/Projects/QA_Vietnamese_News/src/api/model/total_output_clean.pkl'
with open(file_path, 'rb') as file:
    total_output_clean = pickle.load(file)

print(total_output_clean[0:10])