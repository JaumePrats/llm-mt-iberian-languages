import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_max_token_length(src_path, tokenizer, return_indexes=False):

    with open(src_path, 'r') as src_file:
        src_lines = [line.strip() for line in src_file.readlines()]
    lengths = []
    for line in tqdm(src_lines):
        tokenized = tokenizer(line).input_ids
        lengths.append(len(tokenized))
    max_lentgh = max(lengths)
    avg_length = sum(lengths) / len(lengths)
    min_length = min(lengths)

    if return_indexes:
        max_index = lengths.index(max_lentgh)
        min_index = lengths.index(min_length)
        return (max_lentgh, max_index), avg_length, (min_length, min_index)
    else:
        return max_lentgh, avg_length, min_length

    tokenized = tokenizer(src_lines[2]).input_ids
    print(tokenized)
    print(len(tokenized))

def main(args):
    src_path = args.path
    model_id = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # tokenized = tokenizer('</s>').input_ids
    # print(len(tokenized))

    max, avg_length, min = get_max_token_length(src_path, tokenizer, return_indexes=True)
    print(src_path)
    print(f'MAX: {max[0]} AVG: {avg_length} MIN: {min[0]}')
    print(f'MAX INDEX: {max[1]}')
    print(f'MIN INDEX: {min[1]}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default='')
    parser.add_argument("--model",type=str, default='tiiuae/falcon-7b') 

    main(parser.parse_args())

