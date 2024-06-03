import argparse
import os
from dataclasses import dataclass

import dask_cudf
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForSeq2SeqLM
import joblib
import gc
from sklearn.linear_model import LinearRegression
import crossfit as cf
from crossfit import op
from crossfit.dataset.home import CF_HOME
from crossfit.backend.torch.hf.model import HFModel
from transformers import AutoConfig, AutoModel, AutoTokenizer
import dask
from tqdm import tqdm
import numpy as np
BATCH_SIZE = 16
NUM_ROWS = 1_000


@dataclass
class Config:
    model = "microsoft/deberta-v3-base"
    fc_dropout = 0.2
    max_len = 512
    out_dim = 3

class CustomModel(nn.Module):
    def __init__(self, config, config_path=None, pretrained=False):
        super().__init__()
        self.config = config
        if config_path is None:
            self.config = AutoConfig.from_pretrained(config.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(config.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        self.fc_dropout = nn.Dropout(config.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, config.out_dim)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def forward(self, batch):
        feature = self.feature(batch["input_ids"], batch["attention_mask"])
        output = self.fc(self.fc_dropout(feature))
        output = torch.softmax(output[:, 0, :], dim=1)
        return output


# The user must provide a load_model function
def load_model(config, device, model_path):
    model = CustomModel(config, config_path=None, pretrained=True)
    model = model.to(device)

    if os.path.exists(model_path):
        sd = torch.load(os.path.join(model_path), map_location="cpu")
        sd = {k[7:] if k.startswith("module.") else k: sd[k] for k in sd.keys()}
        model.load_state_dict(sd, strict=True)

    model.eval()
    return model


class MyModel(HFModel):
    def __init__(self, config):
        self.config = config
        super().__init__(self.config.model)

    def load_model(self, model_path=None, device="cuda"):
        return load_model(self.config, device=device, model_path=model_path or self.path_or_name)

    def load_config(self):
        return AutoConfig.from_pretrained(self.path_or_name)


def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch Model Predictions using Crossfit")
    parser.add_argument("input_parquet_path", help="Input parquet file")
    parser.add_argument("output_parquet_path", help="Output file")
    parser.add_argument(
        "--input-column", type=str, default="text", help="Column name in input dataframe"
    )
    parser.add_argument("--pool-size", type=str, default="12GB", help="RMM pool size")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of GPUs")
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model name",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--partitions", type=int, default=2, help="Number of partitions")

    args = parser.parse_args()
    return args

class AutoModelForSeq2SeqModel(HFModel):
    def load_model(self, device="cuda"):
        #from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForSeq2SeqLM
        if os.path.exists(os.environ.get('HF_DATASETS_CACHE')):
            return AutoModelForSeq2SeqLM.from_pretrained(os.environ.get('HF_DATASETS_CACHE'), trust_remote_code=True,low_cpu_mem_usage=True).to(device)
        else:
            model=AutoModelForSeq2SeqLM.from_pretrained(self.path_or_name, trust_remote_code=True,low_cpu_mem_usage=True)#.to(device)
            model.save_pretrained(os.environ.get('HF_DATASETS_CACHE'))
            return  model.to(device)

    def load_cfg(self):
        return AutoConfig.from_pretrained(self.path_or_name,trust_remote_code=True)
    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.path_or_name,trust_remote_code=True)
    def max_seq_length(self) -> int:
        return self.load_cfg().max_source_positions
    def fit_memory_estimate_curve(self, model=None):
        remove_model = False
        if model is None:
            remove_model = True
            model = self.load_model(device="cuda")

        cache_dir = os.path.join(CF_HOME, "memory", self.load_cfg()._name_or_path)
        mem_model_path = os.path.join(cache_dir, "mem_model.pkl")

        if os.path.exists(mem_model_path):
            self.mem = joblib.load(mem_model_path)
            return self

        print(f"Fitting memory estimate curve for model: {self.path_or_name}")

        device = next(model.parameters()).device
        X = []
        y = []

        max_seq = self.max_seq_length()
        #for batch_size in tqdm(range(2048, 0, -256)):
        for batch_size in tqdm(range(16, 0, -2)):
            if batch_size <= 0:
                continue

            for seq_len in range(max_seq, 0, -64):
                if seq_len <= 0:
                    continue

                torch.cuda.reset_peak_memory_stats()
                
                #batch_encoding = BatchEncoding.from_dict(tokenized_dict)
                batch = {
                   "input_ids": torch.randint(1, 501, (batch_size, seq_len)).to(device=device),
                   "attention_mask": torch.ones((batch_size, seq_len)).to(device=device),
                }
                try:
                    _ = model.generate(
                            **batch,
                            use_cache=True,
                            min_length=0,
                            max_length=256,
                            num_beams=5,
                            num_return_sequences=1
                                      )
                    memory_used = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB
                    X.append([batch_size, seq_len, seq_len**2])
                    y.append(memory_used)

                except RuntimeError as e:
                    if "out of memory" in str(e) or "out_of_memory" in str(e):
                        pass
                    else:
                        raise e
                finally:
                    del batch
                    if "outputs" in vars():
                        del outputs
                    gc.collect()
                    torch.cuda.empty_cache()

        self.mem = LinearRegression().fit(np.array(X), np.array(y))
        os.makedirs(cache_dir, exist_ok=True)
        joblib.dump(self.mem, mem_model_path)

        if remove_model:
            del model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    args = parse_arguments()

    ddf = dask_cudf.read_parquet(args.input_parquet_path)

    labels = ["foo", "bar", "baz"]
    model_name = "ai4bharat/indictrans2-en-indic-1B"
    model=AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True,low_cpu_mem_usage=True)
    model.save_pretrained(os.environ.get('HF_DATASETS_CACHE'))
    print(f'==>> Model saving complete...')
    with cf.Distributed(rmm_pool_size=args.pool_size, n_workers=args.num_workers):
        model_name = "ai4bharat/indictrans2-en-indic-1B"
        model = AutoModelForSeq2SeqModel(model_name)
        pipe = op.Sequential(
            op.Tokenizer(model, cols=[args.input_column], tokenizer_type="sentencepiece"),
            op.Predictor(model, sorted_data_loader=True, batch_size=args.batch_size),
            op.Labeler(labels, cols=["preds"]),
            repartition=args.partitions,
            keep_cols=[args.input_column],
        )
        outputs = pipe(ddf)
        outputs.to_parquet(args.output_parquet_path)


if __name__ == "__main__":
    main()

