import os
import pandas as pd
from tqdm import tqdm
import gc,time
import glob,json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

cpu_input_dir = "/lustre/fsw/portfolios/llmservice/users/uahmed/orig_inputs/very_large_files/demo/cpu_inp1"
next_cpu_inp_dir = "/lustre/fsw/portfolios/llmservice/users/uahmed/temp_out/temp_rem_inp"
cpu_out_dir = "/lustre/fsw/portfolios/llmservice/users/uahmed/orig_inputs/very_large_files/demo/cpu_out1"
group_column = 'adlr_id'
def read_file(file_path):
    return pd.read_json(file_path, lines=True) 

def single_file_proc(file):
    # print()
    df_comp = pd.read_json(file, lines=True)
    df_comp["filename"] = os.path.basename(file)
    df = df_comp[[group_column,"num_sen","filename"]]
    grouped = df.groupby(group_column)
    non_completed_singlefile_ids = []
    completed = []
    overwrite = False
    for group_value, group_df in grouped:
        if len(group_df)==group_df.iloc[0]['num_sen']:
            completed.append(group_value)
        else:
            overwrite = True
            non_completed_singlefile_ids.append(group_value)
    if overwrite:
        print(f"partial ids : {non_completed_singlefile_ids}")
        df_comp = df_comp[~df_comp[group_column].isin(non_completed_singlefile_ids)]
        output_filename = os.path.join(cpu_out_dir, f"{os.path.basename(file)}")
        df_comp.to_json(output_filename, orient='records', lines=True)
    return completed
        
def input_file_filtering(file, ids_list):
    print(f"got list of len : {len(ids_list)}")
    df = pd.read_json(file, lines=True)
    remaining_df = df[~df[group_column].isin(ids_list)]
    print(f"len remaining df for file {file}: {len(remaining_df)}")
    if len(remaining_df)!=0:
        output_filename = os.path.join(next_cpu_inp_dir, f"{os.path.basename(file)}")
        print(f"Writing reamining of {file} to {output_filename}")
        remaining_df.to_json(output_filename, orient='records', lines=True, force_ascii=False) 
    return True




all_data = pd.DataFrame()

completed_ids = []
non_completed_ids = []
bs = 50
cpu_out_files = os.listdir(cpu_out_dir)

if __name__ == "__main__":
    st = time.time()
    jsonl_files_1 = [os.path.join(cpu_out_dir, f) for f in os.listdir(cpu_out_dir) if f.endswith('.jsonl')]
    all_comp_ids=[]
    print(f"here")
    with multiprocessing.Pool(processes=50) as pool1:
        all_comp_ids = pool1.starmap(single_file_proc, zip(jsonl_files_1))
    del pool1
    print(f"Got len of completed ids = {len(all_comp_ids)}")
    all_ids = []
    for idx,l in enumerate(all_comp_ids):
        # print(f"len of {idx}:{len(l)}")
        all_ids.extend(l)
    print(f"All ids : {len(all_ids)}")
    print(f"Time to get all completed = {time.time()-st} sec")
    jsonl_files_2 = [os.path.join(cpu_input_dir, f) for f in os.listdir(cpu_input_dir) if f.endswith('.jsonl')]
    inp_l=[]
    for i in range(len(jsonl_files_2)):
        inp_l.append(all_ids)
    with multiprocessing.Pool(processes=10) as pool2:
        print(f"reached here")
        res = pool2.starmap(input_file_filtering, zip(jsonl_files_2, inp_l))
    print(f"complete")
    

# for i in tqdm(range(0, len(cpu_out_files), bs), total = int(len(cpu_out_files)/bs)+1):
#     df_comp = pd.DataFrame()
#     comp_lis = []
#     for filename in tqdm(cpu_out_files[i:i+bs], total=bs):

#     # if filename.endswith(".jsonl"):
#         if filename.endswith(".part"):
#             file_path = os.path.join(cpu_out_dir, filename)
#             # comp_lis.append(file_path)
#             with open(file_path) as f:
#                 data=f.readlines()
#             comp_lis.extend([json.loads(d.strip()) for d in data])
#             # sfpd = pd.read_json(file_path, lines=True)
#             # sfpd["filename"]=filename
#             # do this once per 50 files
#             # df_comp = pd.concat([df_comp, sfpd])
#             # comp_lis.append(sfpd)
#     # with ProcessPoolExecutor() as executor:
#     #     # Read files in parallel
#     #     dfs = list(executor.map(read_file, comp_lis))    
#     # df_comp = pd.concat(dfs)#, lines=True)
#     df_copm = pd.DataFrame(comp_lis)
#     print(df_comp.head())
#     df = df_comp[[group_column,"num_sen","filename"]]
#     grouped = df.groupby(group_column)
#     non_completed_singlefile_ids = []
#     # id_belongs_to = []
#     overwrite = False
#     for group_value, group_df in grouped:
#         if len(group_df)==group_df.iloc[0]['num_sen']:
#             completed_ids.append(group_value)
#         else:
#             overwrite = True
#             non_completed_singlefile_ids.append(group_value)
#             non_completed_ids.append(group_value)
#             # id_belongs_to.append(group_df.iloc[0]["filename"])
#     if overwrite:
#         # for id,file in zip(non_completed_singlefile_ids, id_belongs_to):
#         #     df_compl[df_compl["filename"]==file & ]
#         df_comp = df_comp[~df_comp[group_column].isin(non_completed_singlefile_ids)]
        
#         # else:
#     #     print(f"No doc in {filename} is partially processed.")
#     del df
#     del df_comp
#     gc.collect()
# print(f"Partial ids : {len(non_completed_ids)}")
# print(f"Completed ids : {len(completed_ids)}")
# print("Partial processing done!")

# print("Going to copy remaining docs from input to another directory which will be input for next cpu cluster")

# completed_df = pd.DataFrame(completed_ids, columns=['adlr_id'])
# cpu_input_dir = "/lustre/fsw/portfolios/llmservice/users/uahmed/orig_inputs/very_large_files/inp"
# next_cpu_inp_dir = "/lustre/fsw/portfolios/llmservice/users/uahmed/temp_out/temp_rem_inp"
# for filename in tqdm(os.listdir(cpu_input_dir), total=len(os.listdir(cpu_input_dir))):
#     if filename.endswith("jsonl"):
#         print(f"Reading file : {filename}")
#         file_path = os.path.join(cpu_input_dir, filename)
#         df = pd.read_json(file_path, lines=True)
#         remaining_df = df[~df["adlr_id"].isin(completed_df["adlr_id"])]
#         print(f"len remaining df for file {filename}: {len(remaining_df)}")
#         if len(remaining_df)!=0:
#             output_filename = os.path.join(next_cpu_inp_dir, f"{filename}")
#             # remaining_df.to_json(output_filename, orient='records', lines=True, force_ascii=False)
#         del df
#         del remaining_df
#     gc.collect()
 



# gpu partital files logic
# gpu_input_dir = "/lustre/fsw/portfolios/llmservice/users/uahmed/orig_inputs/very_large_files/out_cpu_ps20w36"
# gpu_output_dir = "/lustre/fsw/portfolios/llmservice/users/uahmed/orig_inputs/very_large_files/gpu_out1"
# output_file_processed = 0
# partial_processed = 0
# for filename in tqdm(os.listdir(gpu_output_dir), total=len(os.listdir(gpu_output_dir))):
#     print(f"reading file : {filename}")
#     if filename.endswith(".jsonl"):
#         output_file_processed += 1
#         file_path = os.path.join(gpu_output_dir, filename)
#         df = pd.read_json(file_path, lines=True)
#         uniq_values = df[group_column].unique()
#         input_file_path = os.path.join(gpu_input_dir, filename)
#         df_inp = pd.read_json(input_file_path, lines=True)
#         df_inp_rem = df_inp[~df_inp[group_column].isin(uniq_values)]

#         if len(df_inp_rem) != 0:
#             partial_processed += 1
#             print(f"Remaiinig of file {filename} is writing...")
#             print(f"len of rem : {len(df_inp_rem)}")
#             df_inp_done = df_inp[df_inp[group_column].isin(uniq_values)]
#             print(f"len of done : {len(df_inp_done)}")
#             # check before writing
#             # df_inp_done.to_json(input_file_path, orient='records', lines=True, force_ascii=False)
#             # df_inp_rem.to_json(input_file_path[:-6]+'_remaining.jsonl', orient='records', lines=True, force_ascii=False)
#         else:
#             print(f"Nothing is remining of file : {filename}")

# jsonl_files = [f for f in os.listdir(gpu_input_dir) if f.endswith('.jsonl')]
# input_files = len(jsonl_files)
# print(f"Total input files : {input_files}")
# print(f"Output files checked : {output_file_processed}")
# print(f"Partial files created : {partial_processed}")
