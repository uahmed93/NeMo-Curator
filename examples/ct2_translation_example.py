import os

os.environ["DASK_DATAFRAME__QUERY_PLANNING"] = "False"
import argparse
import re
import time
from dataclasses import dataclass
from functools import lru_cache

import cudf
import pandas as pd
import ctranslate2
import numpy as np
import torch
import torch.nn as nn
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel
from dask.distributed import get_worker
from nltk.tokenize import sent_tokenize
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from nemo_curator.classifiers.base import DistributedDataClassifier
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client, load_object_on_worker
from nemo_curator.utils.script_utils import ArgumentHelper

try:
    from IndicTransToolkit import IndicProcessor
except ImportError:
    raise ImportError(
        "IndicTransToolkit not found. Please install it using the following command: \n"
        + "pip install git+https://github.com/VarunGumma/IndicTransToolkit.git"
    )

TERMINAL_PUNCTUATIONS = (
    ".",
    "!",
    "?",
    ":",
    ",",
    ";",
    ")",
    "}",
    "]",
    '"',
    "'",
    "”",
    "’",
)
START_PUNCTUATIONS = ("(", "{", "[", "'", '"', "“", "‘")


@dataclass
class TranslationConfig:
    pretrained_model_name_or_path: str
    ct2_model_path: str
    max_words_per_sen: int = 200
    target_lang_code: str = "hin_Deva"

class CT2CustomModel():
    def __init__(self, config: TranslationConfig, device="cuda"):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
            trust_remote_code = True,
        )
        self.model = ctranslate2.Translator(model_path=config.ct2_model_path, device=device)

    def clean_extra_tokens(self, token_2d):
        results=[]
        for token_1d in token_2d:
            result = []
            for t in token_1d:
                if t==self.tokenizer.pad_token or t==self.tokenizer.bos_token or t==self.tokenizer.eos_token or t==self.tokenizer.unk_token:
                    pass
                else:
                    result.append(t)
            results.append(result)
        return results

    def __call__(self, batch):
        token_ids_2d=batch['input_ids']
        token_ids_1d = token_ids_2d.view(-1).tolist()
        tokens_1d = self.tokenizer.convert_ids_to_tokens(token_ids_1d)
        tokens_2d = [tokens_1d[i:i + token_ids_2d.size(1)] for i in range(0, len(tokens_1d), token_ids_2d.size(1))]
        tokenss = self.clean_extra_tokens(tokens_2d)

        tr_res = self.model.translate_batch(
            tokenss,
            min_decoding_length=0,
            max_decoding_length=256,
            beam_size=5,
            num_hypotheses=1,
            repetition_penalty=1.02 ######
        )
        translations = ["".join(x.hypotheses[0]) for x in tr_res]
        return translations

class ModelForSeq2SeqModel(HFModel):
    def __init__(self, config):
        self.trans_config = config
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.trans_config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )
        # self.string_tok_inf = config.string_tok_inf
        super().__init__(self.trans_config.pretrained_model_name_or_path, model_output_type="string")

    def load_model(self, device="cuda"):
        model = CT2CustomModel(
            self.trans_config
        )
        return model

    def load_config(self):
        return AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.trans_config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )

    @lru_cache(maxsize=1)
    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.trans_config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )

    def max_seq_length(self) -> int:
        return self.config.max_source_positions

    @lru_cache(maxsize=1)
    def load_cfg(self):
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.trans_config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )
        return config


class IndicTranslation(DistributedDataClassifier):
    def __init__(
        self,
        ct2_model_path: str,
        pretrained_model_name_or_path: str = "ai4bharat/indictrans2-en-indic-1B",
        input_column: str = "indic_proc_text",
        batch_size: int = 128,
        autocast: bool = False,
        target_lang_code: str = "hin_Deva"
    ):
        # self.ct2_model_path = ct2_model_path
        # self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.input_column = input_column
        self.batch_size = batch_size
        self.autocast = autocast

        self.translation_config = TranslationConfig(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            ct2_model_path=ct2_model_path,
            target_lang_code=target_lang_code
        )
        self.model = ModelForSeq2SeqModel(self.translation_config)
        super().__init__(
            model=self.model,
            batch_size=self.batch_size,
            device_type="cuda",
            autocast=self.autocast,
            labels=None,
            filter_by=None,
            out_dim=None,
            pred_column=None,
            max_chars=None,
        )

    def preprocess_df(self, df: cudf.DataFrame) -> cudf.DataFrame:
        ip = load_object_on_worker(
            "IndicProcessor", IndicProcessor, {"inference": True}
        )
        indices = df["text"].index.to_arrow().to_pylist()
        sentences = df["text"].to_arrow().to_pylist()
        sentences = ip.preprocess_batch(
            sentences, src_lang="eng_Latn", tgt_lang=self.translation_config.target_lang_code#"hin_Deva"
        )
        df["indic_proc_text"] = cudf.Series(sentences, index=indices)
        return df

    def translate_tokens(self, df: cudf.DataFrame) -> cudf.DataFrame:
        worker = get_worker()
        if hasattr(worker, "IndicProcessor"):
            ip = getattr(worker, "IndicProcessor")
        else:
            ip = load_object_on_worker(
                "IndicProcessor", IndicProcessor, {"inference": True}
            )
        tokenizer = self.model.load_tokenizer()
        indices = df["translation"].index.to_arrow().to_pylist()
        generated_tokens = df["translation"].to_arrow().to_pylist()
        converted_tokens = []
        for g in generated_tokens:
            converted_tokens.append(tokenizer.convert_tokens_to_string(g))
        converted_tokens = ip.postprocess_batch(converted_tokens, lang=self.translation_config.target_lang_code)#"hin_Deva")
        df["translation"] = cudf.Series(data=converted_tokens,index=indices)
        return df

    def has_alphabet_characters(self, text: str) -> bool:
        return any(c.isalpha() for c in text)

    def custom_tokenize(self, text: str):
        split_text = re.split(
            r"(\#{2,}|\_{2,}|\…{2,}|\+{2,}|\.{2,}|\-{3,}|\*{2,}|\~{2,}|\={2,}|\!{2,}|\n|\t|\‣|\⁃|\⁌|\⁍|\●|\○|\•|\·|\◘|\◦|\⦾|\⦿|\|)",
            text,
        )
        split_text = [s for s in split_text if len(s) > 0]
        tokenized_sentences = []
        len_flag = False
        for line in split_text:
            # Tokenize sentences using NLTK's sent_tokenize function
            if self.has_alphabet_characters(line) == True:
                sentences = sent_tokenize(line)
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
            if len(sent) <= self.translation_config.max_words_per_sen:
                tokenized_sentence_len.append(sentence)

        return tokenized_sentence_len

    def truncate_start_end_symbols(self, input_string):
        start = ""
        mid = ""
        end = ""
        flag = True
        for char in input_string:
            if char.isalnum() == False:
                if flag:
                    start += char
                else:
                    end += char
            else:
                flag = False
                end = ""
        mid = input_string[len(start) : len(input_string) - len(end)]
        while len(start):
            if start[-1] in START_PUNCTUATIONS:
                mid = start[-1] + mid
                start = start[: len(start) - 1]
            else:
                break
        while len(end):
            if end[0] in TERMINAL_PUNCTUATIONS:
                mid += end[0]
                end = end[1:]
            else:
                break
        return pd.Series([start, mid, end], index=['start_sym', 'text', 'end_sym'])
    

    def process_input_text(self, df: cudf.DataFrame) -> cudf.DataFrame:
        df = df.to_pandas()
        df["text"] = df["text"].apply(self.custom_tokenize)
        df["doc_id"] = np.arange(1, len(df) + 1)
        df = df.explode("text", ignore_index=True)
        # df = df.reset_index(drop=False)
        df = df.reset_index(drop=True)

        df["start_sym"] = ""
        df["end_sym"] = ""
        df["has_letters"] = df["text"].apply(lambda x:self.has_alphabet_characters(x))

        df.loc[df["has_letters"],['start_sym', 'text', 'end_sym']] = df.loc[df["has_letters"],'text'].apply(self.truncate_start_end_symbols)#, result_type='expand')

        df = cudf.DataFrame.from_pandas(df)
        return df

    def remove_false_fullstop(self, df: cudf.DataFrame) -> cudf.DataFrame:
        if self.translation_config.target_lang_code == "hin_Deva":
            lang_full_stop = "|"
        else:
            lang_full_stop = "."
        engligh_stop_flag = df["text"].str.endswith(".")
        # hindi_stop_flag = df["translation"].str.endswith("|")
        hindi_stop_flag = df["translation"].str.endswith(lang_full_stop)
        # df["translation"][~engligh_stop_flag & hindi_stop_flag] = df[
        #     "translation"
        # ].str.rstrip("|")
        df["translation"][~engligh_stop_flag & hindi_stop_flag] = df[
            "translation"
        ].str.rstrip(lang_full_stop)

        df["translation"] = df["translation"].str.strip()
        return df

    def grouping(self, df: cudf.DataFrame) -> cudf.DataFrame:
        df = df.to_pandas()
        print(f"Coming here for GROUPING ,cols: {df.columns}")
        agg_funcs = {
            "translation": lambda s: "".join(s),
            "text": lambda s: "".join(s),
            "len_filter": 'all',
            # "indic_proc_text": lambda s: "".join(s),
        }
        other_columns = {
            col: "first"
            for col in df.columns
            if col not in agg_funcs and col != "doc_id"
        }

        agg_funcs.update(other_columns)
        df = df.groupby("doc_id").agg(agg_funcs).reset_index()
        df = cudf.DataFrame.from_pandas(df)
        return df


    def _run_classifier(self, dataset: DocumentDataset) -> DocumentDataset:
        ddf = dataset.df

        ddf_all = ddf[['text','filename']]

        ddf_meta = ddf_all._meta.copy()

        print(f"columns = {ddf_all.columns.tolist()}")
        ddf_meta['doc_id'] = ""
        ddf_meta['start_sym'] = ""
        ddf_meta['end_sym'] = ""
        ddf_meta['has_letters'] = False
        print(f"meta columns = {ddf_meta.columns.tolist()}")
        
        ddf_all = ddf_all.map_partitions(self.process_input_text, meta=ddf_meta)

        ##### internal
        ddf = ddf_all[ddf_all["has_letters"]]
        ddf_false = ddf_all[~ddf_all["has_letters"]]
        # ddf false opn
        ddf_false["translation"] = ddf_false["text"]

        preprocessed_meta = ddf_meta.copy()
        preprocessed_meta['indic_proc_text'] = ''
        ddf = ddf.map_partitions(self.preprocess_df,meta=preprocessed_meta)#ddf_meta)# meta=ddf._meta.copy())

        columns = ddf.columns.tolist()
        self.model = ModelForSeq2SeqModel(self.translation_config)

        pipe = op.Sequential(
            op.Tokenizer(
                self.model, cols=[self.input_column], tokenizer_type="default", max_length=255
            ),
            op.Predictor(
                self.model,
                sorted_data_loader=True,
                batch_size=self.batch_size,
                pred_output_col="translation",
            ),
            keep_cols=columns,
        )
        ddf = pipe(ddf)
        translated_meta = ddf._meta.copy()
        translated_meta["translation"] = "DUMMY_STRING"

        ddf = ddf.map_partitions(self.translate_tokens, meta=translated_meta)
        
        # ddf["indic_proc_text"] = ddf["start_sym"] + ddf["indic_proc_text"] + ddf["end_sym"]
        ddf["text"] = ddf["start_sym"] + ddf["text"] + ddf["end_sym"]
        en_len = ddf["text"].str.len()
        tr_len = ddf["translation"].str.len()
        # ddf["len_filter"]=False
        ddf["len_filter"] = tr_len <= 5*en_len
        # ddf = ddf[tr_len <= 5*en_len]
        
        ddf = ddf.map_partitions(self.remove_false_fullstop, meta=ddf._meta.copy())
        # Add start and end symbols :
        ddf["translation"] = ddf["start_sym"] + ddf["translation"] + ddf["end_sym"]
        # ddf["indic_proc_text"] = ddf["start_sym"] + ddf["indic_proc_text"] + ddf["end_sym"]

        # if you want grouping sentences as docs
        # ddf_all['false_translation'] = ddf_false['translation']
        # ddf_all['false_translation']=ddf_all['false_translation'].fillna('')
        # ddf_all['translation'] = ddf['translation']
        # ddf_all['translation']=ddf_all['translation'].fillna('')
        # ddf_all['translation'] = ddf_all['translation']+ddf_all['false_translation']
        
        # ddf_all['len_filter']=ddf['len_filter']
        # ddf_all['len_filter']=ddf_all['len_filter'].fillna(True)
        
        # ddf_all = ddf_all.drop(columns=["false_translation", "start_sym", "end_sym", "has_letters"])
        # ddf_all = ddf_all.map_partitions(self.grouping)#, meta=ddf._meta)
        

        # return DocumentDataset(ddf_all)
        return DocumentDataset(ddf)


def attach_args():
    parser = ArgumentHelper.parse_distributed_classifier_args()
    parser.set_defaults(
        pretrained_model_name_or_path="ai4bharat/indictrans2-en-indic-1B"
    )
    parser.set_defaults(input_text_field="text")
    parser.set_defaults(device="gpu")
    return parser


def main(args):
    print(f"Arguments parsed = {args}")
    st = time.time()
    client = get_client(**ArgumentHelper.parse_client_args(args))
    print(client.dashboard_link)
    translator_model = IndicTranslation(
        ct2_model_path=args.ct2_model_path,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        input_column=args.input_text_field,
        batch_size=args.batch_size,
        autocast=args.autocast,
        target_lang_code=args.tgt_lang
    )
    input_files = [
        os.path.join(args.input_data_dir, x) for x in os.listdir(args.input_data_dir)
    ]
    input_dataset = DocumentDataset.read_json(
        input_files, backend="cudf", add_filename=True
    )
    result_dataset = translator_model(dataset=input_dataset)

    result_dataset.to_json(output_file_dir=args.output_data_dir, write_to_filename=True)
    print(f"Total time taken for translation: {time.time()-st} seconds", flush=True)
    client.close()


if __name__ == "__main__":
    parser = attach_args()
    parser.add_argument('--ct2-model-path',type=str,required=True,help="CT2 Model directory")
    parser.add_argument('--tgt-lang',default="hin_Deva",type=str,help="Language code for which translation will run")
    main(parser.parse_args())

