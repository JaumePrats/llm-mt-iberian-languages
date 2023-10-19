#!/bin/bash

src_data='/fs/surtr0/jprats/data/raw/flores200_dataset/dev/eng_Latn.dev'
out_data='test_flores_tr.es'

python /fs/alvis0/jprats/code/llm-mt-iberian-languages/test_aguila.py < $src_data > $out_data