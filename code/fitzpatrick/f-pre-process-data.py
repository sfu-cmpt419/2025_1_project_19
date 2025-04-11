import pandas as pd

input_file = "fitzpatrick17k.csv"
output_file = "fitzpatrick-preprocessed.csv"
input_dir = "../../datasets/fitzpatrick/"
output_dir = "../../datasets/fitzpatrick/"

# 1. Read the CSV file
df = pd.read_csv(input_dir + input_file)

# 2. Remove the specified column
df = df.drop(columns=[
    'fitzpatrick_centaur',
    'label',
    'nine_partition_label',
    'three_partition_label',
    'qc',
    'url',
    'url_alphanum'
])

# 3. Rename columns
df = df.rename(columns={
    'md5hash': 'image_hash',
    'fitzpatrick_scale': 'fitzpatrick_scale'
})

# 4. Remove rows where fitzpatrick_scale == -1
df = df[df['fitzpatrick_scale'] != -1]

# 5. Save the preprocessed file
df.to_csv(output_dir + output_file, index=False)
print(f"📁 [SAVE] Preprocessed file saved to: {output_dir + output_file}")