import random

def sample_lines(input_file, output_file, fraction):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    num_lines = len(lines)
    sample_size = int(fraction * num_lines)
    
    sampled_lines = random.sample(lines, sample_size)
    
    with open(output_file, 'a') as outfile:
        outfile.writelines(sampled_lines)


files = [
    '/fs/surtr0/jprats/code/llm-mt-iberian-languages/src/data/file1.txt', 
    '/fs/surtr0/jprats/code/llm-mt-iberian-languages/src/data/file2.txt'
    ]
out_file = '/fs/surtr0/jprats/code/llm-mt-iberian-languages/src/data/out.txt'

num_lines = []
for file in files:
    print(f'Counting lines in {file} ...')
    with open(file, "rb") as f:
        file_num_lines = sum(1 for _ in f)
    num_lines.append(file_num_lines)

total_num_lines = sum(num_lines)
fractions = [(f_num_lines / total_num_lines) for f_num_lines in num_lines]
for i, file in enumerate(files):
    print(f'FILE: {file}, N.LINES: {num_lines[i]}, FRACTION: {fractions[i]}')

for i, file in enumerate(files):
    print(f'Extracting lines from {file}')
    sample_lines(file, out_file, fractions[i])

import linecache

def get_specific_line(file_path, line_number):
    line = linecache.getline(file_path, line_number)
    return line.strip()  # Remove newline characters