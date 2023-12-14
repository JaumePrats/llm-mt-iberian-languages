import pandas as pd

# Read the CSV file into a DataFrame
csv_file_path = '/fs/surtr0/jprats/data/raw/idioms_francesca/idioms_sentences.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(csv_file_path)

# # Extract each column into a different text file
# for column in df.columns:
#     # Create a new text file for each column
#     txt_file_path = f'/fs/surtr0/jprats/data/raw/idioms_francesca/{column}.txt'

#     # Write the column values to the text file
#     with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
#         df[column].to_string(txt_file, index=False)

#     print(f"Extracted '{column}' column to '{txt_file_path}'")

all_en_path = '/fs/surtr0/jprats/data/raw/idioms_francesca/extraction/idioms_francesca.all.eng'
all_es_path = '/fs/surtr0/jprats/data/raw/idioms_francesca/extraction/idioms_francesca.all.spa'
idioms_en_path = '/fs/surtr0/jprats/data/raw/idioms_francesca/extraction/idioms_francesca.idioms.eng'
idioms_es_path = '/fs/surtr0/jprats/data/raw/idioms_francesca/extraction/idioms_francesca.idioms.spa'
distr_en_path = '/fs/surtr0/jprats/data/raw/idioms_francesca/extraction/idioms_francesca.distractors.eng'
distr_es_path = '/fs/surtr0/jprats/data/raw/idioms_francesca/extraction/idioms_francesca.distractors.spa'

with open(all_en_path, 'w') as all_en_file, open(all_es_path, 'w') as all_es_file, open(idioms_en_path, 'w') as idioms_en_file, open(idioms_es_path, 'w') as idioms_es_file, open(distr_en_path, 'w') as distr_en_file, open(distr_es_path, 'w') as distr_es_file:

    for index, row in df.iterrows():
        sentence_id = row['sentence_id']
        idiom_key = row['idiom_key']
        en = row['en']
        es = row['es']

        all_en_file.write(en + '\n')
        all_es_file.write(es + '\n')

        if sentence_id.split('-')[-1] == 'di': # distractor sentence
            distr_en_file.write(en + '\n')
            distr_es_file.write(es + '\n')
        else:
            idioms_en_file.write(en + '\n')
            idioms_es_file.write(es + '\n')



