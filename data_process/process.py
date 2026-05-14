import pandas as pd
import os
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import json
from tqdm import tqdm
import random


def split_to_parquet(df, rows_per_file, base_path="output"):
    n = len(df)
    for i in range(0, n, rows_per_file):
        df_part = df.iloc[i:i+rows_per_file]
        file_path = f"{base_path}/part_{i//rows_per_file}.parquet"
        df_part.to_parquet(file_path, index=False)
        print("Saved:", file_path)


def flatten_dataset(df):
    """
    Flatten the dataset such that each question-answer pair becomes a single item.
    Returns:
        flattened_data (list): List of dictionaries with image data and each QA pair.
    """
    flattened_data = []

    for idx, row in df.iterrows():
        # Extract the bytes from the 'image' dictionary
        image_data = row['image']  # Access the image bytes
        # Convert the image bytes to a PIL Image
        # try:
        #     image = Image.open(BytesIO(image_data)).convert("RGB")
        # except Exception as e:
        #     print(f"Error loading image at index {idx}: {e}")
        #     continue

        # Safely load metadata as JSON
        try:
            metadata = json.loads(row['metadata'])  # Using json.loads to parse JSON safely
        except json.JSONDecodeError as e:
            print(f"Error decoding metadata at index {idx}: {e}")
            continue
        for qa_pair in metadata:
            question = qa_pair.get("Question", "")
            answer = qa_pair.get("Answer", "")

            if question and answer:
                flattened_data.append({
                    "images": image_data,
                    "problem": question,
                    "answer": answer,
                })
    # print(flattened_data)
    return flattened_data

# parquet_file = pd.read_parquet('/home/guangjiahui/guangjiahui/MMUnlearner-main/data_split/baseline_train_split_flatten/mix_10_90_2000')
# print(parquet_file.iloc[0].to_dict())
# exit()

data_files = os.listdir('./baseline_train_split')

# for dir in data_files:
#     if '_' in dir:
#         parquet_file = pd.read_parquet(f'./baseline_train_split/{dir}/train-00000-of-00001.parquet')
#         flatten_data = flatten_dataset(parquet_file)
#         flatten_parquet_file = pd.DataFrame(flatten_data)
#         os.makedirs(f'./baseline_train_split_flatten/{dir}', exist_ok=True)
#         # flatten_parquet_file.to_parquet(f'./baseline_train_split_flatten/{dir}/train-00000-of-00001.parquet')
#         split_to_parquet(flatten_parquet_file, 600, base_path=f'./baseline_train_split_flatten/{dir}')
#         # print(parquet_file.iloc[0])
#         # image = parquet_file.iloc[0]['images']
#         # image = Image.open(BytesIO(image)).convert("RGB")
#         # image.save('./test.png')
#         # exit()

forget_df = pd.read_parquet('/home/guangjiahui/guangjiahui/MMUnlearner-main/data_split/baseline_train_split/forget_5/train-00000-of-00001.parquet')
forget_data = flatten_dataset(forget_df)
# random.shuffle(forget_data)
retain_df = pd.read_parquet('/home/guangjiahui/guangjiahui/MMUnlearner-main/data_split/baseline_train_split/retain_85/train-00000-of-00001.parquet')
retain_data = flatten_dataset(retain_df)
# random.shuffle(retain_data)
# retain_data_1000 = random.sample(retain_data, len(forget_data))
# print(forget_data[0])
# print(type(forget_data[0]['images']))
# print(forget_data[0]['images'].keys())
# exit()

all_data = []
# num_forget = len(forget_data)
# idx = 0
# for rd in retain_data_1000:
#     all_data.append(rd)
#     all_data.append(forget_data[idx % num_forget])
#     idx += 1

# print(all_data)
# print(len(all_data))


# for fd in tqdm(forget_data):
#     forget_question = fd['problem']
#     forget_image = fd['images']
#     forget_answer = fd['answer']
#     count = 0
#     for rd in retain_data:
#         retain_question = rd['problem']
#         retain_image = rd['images']
#         retain_answer = rd['answer']
#         if count % 2 == 0:
#             all_data.append({
#                 "images": [forget_image, retain_image],
#                 "problem": f"Answer the following two questions in one sentence each and output the answers in the LIST format.\n<image>\n{forget_question}, \n<image>\n{retain_question}",
#                 "answer": f"{forget_answer}||{retain_answer}",
#                 "forget_first": True
#             })
#         else:
#             all_data.append({
#                 "images": [retain_image, forget_image],
#                 "problem": f"Answer the following two questions in one sentence each and output the answers in the LIST format.\n<image>\n{retain_question}, \n<image>\n{forget_question}",
#                 "answer": f"{retain_answer}||{forget_answer}",
#                 "forget_first": False
#             })
#         count += 1


forget_data = random.sample(forget_data, 200)
retain_data = random.sample(retain_data, 200)
print(len(forget_data), len(retain_data))
# for count, (fd, rd) in tqdm(enumerate(zip(forget_data, retain_data))):
#     forget_question = fd['problem']
#     forget_image = fd['images']
#     forget_answer = fd['answer']
#     retain_question = rd['problem']
#     retain_image = rd['images']
#     retain_answer = rd['answer']
    # if count % 2 == 0:
    #     all_data.append({
    #         "images": [forget_image, retain_image],
    #         "problem": f"Answer the following two questions in one sentence each and output the answers in the LIST format.\n<image>\n{forget_question}\n<image>\n{retain_question}",
    #         "answer": f"{forget_answer}||{retain_answer}",
    #         "forget_first": True
    #     })
    # else:
    #     all_data.append({
    #         "images": [retain_image, forget_image],
    #         "problem": f"Answer the following two questions in one sentence each and output the answers in the LIST format.\n<image>\n{retain_question}\n<image>\n{forget_question}",
    #         "answer": f"{retain_answer}||{forget_answer}",
    #         "forget_first": False
    #     })
for fd in tqdm(forget_data):
    forget_question = fd['problem']
    forget_image = fd['images']
    forget_answer = fd['answer']
    all_data.append({
        "images": [forget_image],
        "problem": "<image>\n" + forget_question,
        "answer": forget_answer,
        "forget_first": True
    })
for rd in tqdm(retain_data):
    retain_question = rd['problem']
    retain_image = rd['images']
    retain_answer = rd['answer']
    all_data.append({
        "images": [retain_image],
        "problem": "<image>\n" + retain_question,
        "answer": retain_answer,
        "forget_first": False
    })

# all_data = random.sample(all_data, 2000)
print(len(all_data))
flatten_parquet_file = pd.DataFrame(all_data)
os.makedirs(f'./baseline_train_split_flatten/mix_separate_15_85_400', exist_ok=True)
split_to_parquet(flatten_parquet_file, 600, base_path='./baseline_train_split_flatten/mix_separate_15_85')
