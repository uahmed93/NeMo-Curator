import os
import pandas as pd
from tqdm import tqdm
import gc

#gpu partital files logic
gpu_input_dir = "/lustre/fsw/portfolios/llmservice/users/uahmed/orig_inputs/very_large_files/out_cpu_ps20w36"
gpu_output_dir = "/lustre/fsw/portfolios/llmservice/users/uahmed/orig_inputs/very_large_files/gpu_out1"
group_column = "adlr_id"
output_file_processed = 0
partial_processed = 0
for filename in tqdm(os.listdir(gpu_output_dir), total=len(os.listdir(gpu_output_dir))):
    print(f"reading file : {filename}")
    if filename.endswith(".jsonl"):
        output_file_processed += 1
        file_path = os.path.join(gpu_output_dir, filename)
        df = pd.read_json(file_path, lines=True)
        uniq_values = df[group_column].unique()
        input_file_path = os.path.join(gpu_input_dir, filename)
        df_inp = pd.read_json(input_file_path, lines=True)
        df_inp_rem = df_inp[~df_inp[group_column].isin(uniq_values)]
        inp_rem_l =df_inp_rem[group_column].unique().tolist()
        if len(df_inp_rem) != 0:
            print(f"From inp dir {filename}, ids remaining = {inp_rem_l}")

            partial_processed += 1
            print(f"Remainig of file {filename} is writing...")
            print(f"Total len of inp dir {filename}={len(df_inp)} ")
            print(f"len of rem : {len(df_inp_rem)}")
            df_inp_done = df_inp[df_inp[group_column].isin(uniq_values)]
            print(f"len of done : {len(df_inp_done)}")
            # check before writing
            df_inp_done.to_json(input_file_path, orient='records', lines=True, force_ascii=False)
            df_inp_rem.to_json(input_file_path[:-6]+'_remaining.jsonl', orient='records', lines=True, force_ascii=False)
        else:
            print(f"Nothing is remining of file : {filename}")

jsonl_files = [f for f in os.listdir(gpu_input_dir) if f.endswith('.jsonl')]
input_files = len(jsonl_files)
print(f"Total input files : {input_files}")
print(f"Output files checked : {output_file_processed}")
print(f"Partial files created : {partial_processed}")

