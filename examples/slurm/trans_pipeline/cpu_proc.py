import os
import gc
os.environ["DASK_DATAFRAME__QUERY_PLANNING"] = "False"

import argparse
import re
import time
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
import numpy as np
from dask.distributed import get_worker, wait
from dask.distributed import performance_report
import nltk
from nltk.tokenize import sent_tokenize

from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper
from nemo_curator.utils.file_utils import get_remaining_files
from tqdm import tqdm

def has_alphabet_characters(text: str) -> bool:
    return any(c.isalpha() for c in text)

def custom_tokenize(text: str, tok):
    split_text = re.split(
        r"(\#{2,}|\_{2,}|\…{2,}|\+{2,}|\.{2,}|\-{3,}|\*{2,}|\~{2,}|\={2,}|\!{2,}|\n|\t|\‣|\⁃|\⁌|\⁍|\●|\○|\•|\·|\◘|\◦|\⦾|\⦿|\|)",
        text,
    )
    split_text = [s for s in split_text if len(s) > 0]
    tokenized_sentences = []
    len_flag = False
    for line in split_text:
        # Tokenize sentences using NLTK's sent_tokenize function
        if has_alphabet_characters(line) == True:
            # sentences = sent_tokenize(line)
            sentences = tok.tokenize(line)
            i = 0
            j = 0
            curr_tokenized_snt = []
            non_translation_str = ""
            # Comparing the list of tokenized sentences (using NLTK) and actual sentence and preserving the spaces,
            # newline and other special characters
            while i < len(line):
                if j < len(sentences):
                    stripped_sent = sentences[j].strip()
                    if len(stripped_sent) == 0:
                        j += 1
                        continue
                    # If tokenized sentence matches then moving to next sentence
                    if line[i] == stripped_sent[0]:
                        if non_translation_str != "":
                            curr_tokenized_snt.append(non_translation_str)
                        curr_tokenized_snt.append(stripped_sent)
                        i += len(stripped_sent)
                        j += 1
                        non_translation_str = ""
                    else:
                        non_translation_str += line[i]
                        i += 1
                else:
                    non_translation_str += line[i]
                    i += 1
            if non_translation_str != "":
                curr_tokenized_snt.append(non_translation_str)
            # Add the tokenized sentences to the list
            tokenized_sentences.extend(curr_tokenized_snt)
        else:
            tokenized_sentences.append(line)

    tokenized_sentence_len = []
    for sentence in tokenized_sentences:
        sent = sentence.split()
        # removing the sentences with word length greater than threshold as the model may not be able translate it due to constraint on output token size
        if len(sent) <= 200:#self.translation_config.max_words_per_sen:
            tokenized_sentence_len.append(sentence)

    return tokenized_sentence_len

def process_input_text(df):
    df["orig_file"] = df["filename"]
    tok = nltk.tokenize.PunktTokenizer('english')
    df["text"] = df["text"].apply(lambda x: custom_tokenize(x, tok))
    df["num_sen"] = df["text"].apply(lambda df: len(df))
    df = df.explode("text", ignore_index=True)
    df = df.reset_index(drop=True)
    df["has_letters"] = df["text"].apply(lambda x:has_alphabet_characters(x))

    gc.collect()
    return df

def cpu_process(dataset):
    ddf = dataset.df
    ddf_all = ddf[['text','filename','adlr_id']]

    print(f"columns = {ddf_all.columns.tolist()}")
    ddf_meta = {
        "text": "object",
        "filename": "object",
        "adlr_id": "object",
        "orig_file": "object",
        "num_sen": "int64",
        "has_letters": "bool"
    }
    st = time.time()
    ddf_all = ddf_all.map_partitions(process_input_text, meta=ddf_meta)
    return DocumentDataset(ddf_all)

def attach_args(
        parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        ),
    ):
    parser = ArgumentHelper(parser).add_distributed_args()
    return parser


def main(args):
    print(f"Arguments parsed = {args}")
    st = time.time()
    client = get_client(**ArgumentHelper.parse_client_args(args))

    print(client.dashboard_link)
    
    # if args.input_file_extension is not None:
    #     input_file_extension = args.input_file_extension
    # else:
    #     input_file_extension = args.input_file_type

    with performance_report(filename="/lustre/fsw/portfolios/llmservice/users/uahmed/1cpu-48+12gb-dask-report.html"):
        input_file_extension = "jsonl"
        # input_files = get_remaining_files(
        #     args.input_data_dir, args.output_data_dir, input_file_extension
        # )
        
        input_files = [
            os.path.join(args.input_data_dir, x) for x in os.listdir(args.input_data_dir)
        ]
        st = time.time()
        input_dataset = DocumentDataset.read_json(
            input_files, backend="pandas", add_filename=True
        )
        # l=len(input_dataset.df)
        # rde = time.time()
        # print(f"=====>> Read time : {rde - st}")
        #print(f"{input_dataset.df['adlr_id'].compute()}")
        print(f"TYPE : {type(input_dataset.df)}")
        input_dataset.df = input_dataset.df.repartition(partition_size='20MB')
        input_dataset.df = input_dataset.df.persist()
        wait(input_dataset.df)
        client.rebalance(input_dataset.df)
        # input_dataset.df = input_dataset.df.persist()
        result_dataset = cpu_process(input_dataset)
        result_dataset.to_json(output_file_dir=args.output_data_dir, write_to_filename=False)
        print(f"Total time taken for CPU processing: {time.time()-st} seconds", flush=True)
    client.close()


if __name__ == "__main__":
    parser = attach_args()
    parser.add_argument('--input-data-dir',type=str,required=True,help="Inp dir")
    parser.add_argument('--output-data-dir',type=str,required=True,help="Out dir")
    main(parser.parse_args())

