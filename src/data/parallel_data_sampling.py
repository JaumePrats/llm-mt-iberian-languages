import random

def sample_lines(input_file, output_file, fraction):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    num_lines = len(lines)
    sample_size = int(fraction * num_lines)
    
    sampled_lines = random.sample(lines, sample_size)
    
    with open(output_file, 'a') as outfile:
        outfile.writelines(sampled_lines)


src_files = [
    '/fs/surtr0/jprats/code/llm-mt-iberian-languages/src/data/file1_src.txt', 
    '/fs/surtr0/jprats/code/llm-mt-iberian-languages/src/data/file2_src.txt'
    ]
ref_files = [
    '/fs/surtr0/jprats/code/llm-mt-iberian-languages/src/data/file1_ref.txt', 
    '/fs/surtr0/jprats/code/llm-mt-iberian-languages/src/data/file2_ref.txt'
    ]
output_src_file = '/fs/surtr0/jprats/code/llm-mt-iberian-languages/src/data/out_src.txt'
output_src_file = '/fs/surtr0/jprats/code/llm-mt-iberian-languages/src/data/out_ref.txt'
out_total_lines = 10

src_num_lines = []
for file in src_files:
    print(f'Counting lines in {file}:')
    with open(file, "rb") as f:
        file_num_lines = sum(1 for _ in f)
    print(f'\tlines: {file_num_lines}')
    src_num_lines.append(file_num_lines)

ref_num_lines = []
for file in ref_files:
    print(f'Counting lines in {file}:')
    with open(file, "rb") as f:
        file_num_lines = sum(1 for _ in f)
    print(f'\tlines: {file_num_lines}')
    ref_num_lines.append(file_num_lines)
try:
    assert src_num_lines == ref_num_lines
except AssertionError as e:
    print(120*'*')
    print('ERROR: SOURCE AND REFERENCE NUMBER OF LINES DO NOT MATCH! Check that files are the correct ones and that the order in "src_files" and "ref_files" matches.')
    exit()

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