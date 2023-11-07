from transformers import AutoTokenizer, AutoModelForCausalLM

def get_max_token_length(src_path, tokenizer):

    with open(src_path, 'r') as src_file:
        src_lines = [line.strip() for line in src_file.readlines()]
    lengths = []
    for line in src_lines:
        tokenized = tokenizer(line).input_ids
        lengths.append(len(tokenized))
    max_lentgh = max(lengths)
    avg_length = sum(lengths) / len(lengths)
    min_length = min(lengths)

    return max_lentgh, avg_length, min_length

    tokenized = tokenizer(src_lines[2]).input_ids
    print(tokenized)
    print(len(tokenized))

# src_path = '/fs/surtr0/jprats/data/raw/flores200_dataset/dev/eus_Latn.dev'
# model_id = 'tiiuae/falcon-7b'
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# # tokenized = tokenizer('</s>').input_ids
# # print(len(tokenized))

# max_lentgh, avg_length, min_length = get_max_token_length(src_path, tokenizer)
# print(src_path)
# print(f'MAX: {max_lentgh} AVG: {avg_length} MIN: {min_length}')


