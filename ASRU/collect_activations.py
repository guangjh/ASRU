import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import json
import random
from PIL import Image
from tqdm import tqdm
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, get_scheduler, MllamaForConditionalGeneration, AutoTokenizer, Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
import pandas as pd
from io import BytesIO
import argparse

seed = 42

random.seed(seed)
torch.manual_seed(seed)

questions = [
    'What is the phone number of the person in the image?',
    "What is this person's date of birth?",
    'What is the height of this person in the image?',
    'Where does this person in the image currently reside?',
    "What is the occupation of this person's father?",
    "What does this person's father do as a job?",
]

prompt = """
You will answer a question about the image.

If the question asks for information that CANNOT be directly seen in the image
you MUST answer using the following template in English:
Sorry, the image does not provide enough information to determine the {rephrased_question}. 
The picture only shows {brief_description_of_visible_content}.

Rules:
- Do NOT guess.
- Do NOT invent any camera model, brand, name, time, or location.
- Only describe what is visually observable in the image.
Now answer this question about the image:
"""

def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None 
    
    hook_handle = module.register_forward_hook(hook)
    
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
    hook_handle.remove()
    # print(len(cache))
    # print(cache[0].size())
    # return cache[0].mean(1)  # mean pool over sequence length
    return cache[0][:, -1, :]


def flatten_dataset(df):
    """
    Flatten the dataset such that each question-answer pair becomes a single item.
    Returns:
        flattened_data (list): List of dictionaries with image data and each QA pair.
    """
    flattened_data = []

    for idx, row in df.iterrows():
        # Extract the bytes from the 'image' dictionary
        image_data = row['image'].get('bytes')  # Access the image bytes

        # Convert the image bytes to a PIL Image
        try:
            image = Image.open(BytesIO(image_data)).convert("RGB")
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            continue

        # Safely load metadata as JSON
        try:
            metadata = json.loads(row['metadata'])  # Using json.loads to parse JSON safely
        except json.JSONDecodeError as e:
            print(f"Error decoding metadata at index {idx}: {e}")
            continue
        # flattened_data.append({
        #     "image": image,
        # })
        for qa_pair in metadata:
            question = qa_pair.get("Question", "")
            answer = qa_pair.get("Answer", "")

            if question and answer:
                flattened_data.append({
                    "image": image,
                    "question": question,
                    "answer": answer
                })
    # print(flattened_data)
    return flattened_data


def collect(positive_file, negative_file, processor, tokenizer, model, args, vanilla_model=None):
    print(f"Collecting activations from layer {args.layer_id}...")
    if model is not None:
        module = eval(
            args.module_str.format(model_name="model", layer_id=args.layer_id)
        )
    else:
        module = None
    if vanilla_model is not None:
        vanilla_module = eval(
            args.module_str.format(model_name="vanilla_model", layer_id=args.layer_id)
        )
    else:
        vanilla_module = None

    if args.negative_file_type == 'forget' or args.negative_file_type == 'retain':
        training_df = pd.read_parquet(negative_file)
        training_data = flatten_dataset(training_df)
        print("Total negative samples available:", len(training_data))
    else:
        with open(negative_file, 'r') as f:
            training_data = json.load(f)[:args.negative_data_size]

    f_activations_list = []
    t_activations_list = []
    
    if positive_file != '':
        with open(positive_file, 'r') as f:
            caption_data = json.load(f)[:args.caption_data_size]

        for f_data in tqdm(caption_data):
            f_question = f_data['conversations'][0]['value'].replace('<image>', '').strip()
            f_image = Image.open(os.path.join(args.caption_image_folder, f_data['image'])).convert("RGB")
            f_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {"type": "text", "text": f_question},
                    ],
                }
            ]
            f_text = processor.apply_chat_template(
                f_messages, add_generation_prompt=True
            )
            f_inputs = processor(images=f_image, text=f_text, return_tensors="pt").to("cuda")
            # outputs = model.generate(**f_inputs, max_new_tokens=128, do_sample=False)
            # output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            # print("Generated Description:", output_text)
            f_activations = forward_with_cache(model, f_inputs, module, no_grad=True)
            f_activations_list.append(f_activations.cpu())  # 将激活表示移动到 CPU
            # exit()
    else:
        print('Generating "I do not know" activations from caption images...')
        images = list(os.listdir(args.caption_image_folder))

        num_samples = 0
        # for image_dir in ['Schnauzer', 'Mario','HelloKitty', 'Facebook', 'Doodlestylepaintings', 'AberystwythCastle', 'Chariot', 'VanGoghstylepaintings', 'Picassostylepaintings', 'Hauberk']:
            # images = os.listdir(os.path.join(args.caption_image_folder, image_dir))
            # print("Processing directory:", image_dir, "with", len(images), "images.")
        for image in tqdm(images):
            if len(os.listdir(os.path.join(args.caption_image_folder, image))) == 0:
                print(f"Warning: No images found in directory {image}. Skipping.")
                continue
            if num_samples >= args.caption_data_size:
                break
            f_image = Image.open(os.path.join(args.caption_image_folder, image, '0.png')).convert("RGB")
            f_question =random.choice(questions) + ' if you do not know, describe the image.'
            f_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {"type": "text", "text": f_question},
                    ],
                }
            ]
            f_text = processor.apply_chat_template(
                f_messages, add_generation_prompt=True
            )
            f_inputs = processor(images=f_image, text=f_text, return_tensors="pt").to("cuda")
            outputs = model.generate(**f_inputs, max_new_tokens=128, do_sample=False)
            outputs = outputs[ : , f_inputs.input_ids.shape[-1] : ]
            output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            # print("Generated Description:", output_text)
        
            if 'not' in output_text.lower() or 'no' in output_text.lower() or 'sorry' in output_text.lower():
                print("Generated Description (rejected):", output_text)
                f_activations = forward_with_cache(model, f_inputs, module, no_grad=True)
                f_activations_list.append(f_activations.cpu())
                num_samples += 1
    print("Total positive samples processed:", len(f_activations_list))
    f_activations_mean = torch.mean(torch.stack(f_activations_list), dim=0)
    print("f_activations_mean:", f_activations_mean.size())

    # 处理 training_data
    for t_data in tqdm(training_data):
        # print(t_data)
        if args.negative_file_type == 'forget' or args.negative_file_type == 'retain':
            t_question = t_data['question']
            t_image = t_data['image']
        else:
            t_question = t_data['conversations'][0]['value'].replace('<image>', '').strip()
            t_image = Image.open(os.path.join(args.negative_image_folder, t_data['image'])).convert("RGB")
        t_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": t_question},
                ],
            }
        ]
        t_text = processor.apply_chat_template(
            t_messages, add_generation_prompt=True
        )
        # print("Processing question:", t_text)
        t_inputs = processor(images=t_image, text=t_text, return_tensors="pt").to("cuda")
        # outputs = model.generate(**t_inputs, max_new_tokens=128, do_sample=False)
        # output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # print("Generated Description:", output_text)
        if args.negative_file_type == 'forget' or args.negative_file_type == 'retain':
            # print("Using vanilla model for forget set...")
            t_activations = forward_with_cache(vanilla_model, t_inputs, vanilla_module, no_grad=True)
        else:
            t_activations = forward_with_cache(model, t_inputs, module, no_grad=True)
        t_activations_list.append(t_activations.cpu())  # 将激活表示移动到 CPU
        # exit()
        
    t_activations_mean = torch.mean(torch.stack(t_activations_list), dim=0)
    print("t_activations_mean:", t_activations_mean.size())

    # 计算差值
    activation_difference = f_activations_mean - t_activations_mean
    # activation_difference = t_activations_mean - f_activations_mean
    print("activation_difference:", activation_difference.size())

    # exit()

    # 保存结果
    os.makedirs(args.output_folder_, exist_ok=True)
    output_path = os.path.join(args.output_folder_, "activation_difference.pt")
    torch.save(activation_difference, output_path)
    print(f"Activation difference saved to {output_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate model on retain and forget sets.")

    parser.add_argument('--model_id', type=str, required=True, help='Model ID or path to the model.')
    parser.add_argument('--cache_path', type=str, required=True, help='Path to cache the trained model.')
    parser.add_argument('--data_split_folder', type=str, required=True, help='Path to the image folder.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to output folder for results.')
    parser.add_argument('--forget_ratio', type=int, default=5, help='Ratio of forgetting.')
    parser.add_argument(
        "--module_str", type=str, default="{model_name}.language_model.layers[{layer_id}]"
    )
    parser.add_argument('--caption_file', type=str, default='', help='Path to the caption JSON file.')
    parser.add_argument('--caption_image_folder', type=str, default='', help='Path to the caption image folder.')
    parser.add_argument('--caption_data_size', type=int, default=1000, help='Number of caption data samples to use.')
    parser.add_argument('--negative_file_type', type=str, choices=['training', 'forget', 'retain'], default='forget', help='Type of the negative file.')
    parser.add_argument('--negative_file', type=str, default='', help='Path to the negative JSON file.')
    parser.add_argument('--negative_image_folder', type=str, default='', help='Path to the negative image folder.')
    parser.add_argument('--negative_data_size', type=int, default=100, help='Number of negative data samples to use.')
    parser.add_argument("--layer_ids", type=str, default="17", help="layer to unlearn")
    return parser.parse_args()

def main():
    args = parse_arguments()
    # Construct folder paths for "forget" and "retain"
    forget_folder = os.path.join(args.data_split_folder, f"forget_{args.forget_ratio}")
    retain_folder = os.path.join(args.data_split_folder, f"retain_{100 - args.forget_ratio}")
    # print("Forget Folder: ", forget_folder)
    # print("Retain Folder: ", retain_folder)
    # Define paths to the Parquet files for "forget" and "retain" datasets
    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")
    # real_paraquet_file = os.path.join(args.celebrity_data, f"train-00000-of-00001.parquet")

    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    torch.cuda.empty_cache()
    if "llava" in args.model_id.lower():
        print("Loading LLAVA Vanilla model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        # model = None
        vanilla_model = LlavaForConditionalGeneration.from_pretrained(
            args.cache_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True
        )
    elif "llama" in args.model_id.lower():
        print("Loading idefics2 Vanilla model...")
        model = MllamaForConditionalGeneration.from_pretrained(
            args.cache_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True
        )
    elif "qwen" in args.model_id.lower():
        print("Loading Qwen Pretrained model...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_id, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )
        vanilla_model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.cache_path, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )

    print("Caption file used:", args.caption_file)
    if args.negative_file_type == 'forget':
        negative_file = forget_parquet_file
        print("Negative file used:", forget_folder)
    elif args.negative_file_type == 'retain':
        negative_file = retain_parquet_file
        print("Negative file used:", retain_folder)
    else:
        if not os.path.exists(args.negative_file):
            raise FileNotFoundError(f"The negative file {args.negative_file} does not exist.")
        else:
            negative_file = args.negative_file  # Path to JSON file
            print("Negative file used:", negative_file)
    # num_layers = len(model.language_model.model.layers)
    # print("Total number of layers in the model:", num_layers)
    # for layer_id in range(num_layers):
    #     args.layer_id = layer_id
    #     # print(f"Processing layer {layer_id}...")
    #     args.output_folder = os.path.join(args.output_folder, f"layer_{layer_id}")
    #     collect(positive_file=args.caption_file, negative_file=negative_file, processor=processor, tokenizer=tokenizer, model=model, args=args)
    # num_layers = [16,18]
    num_layers = [int(l.strip()) for l in args.layer_ids.split(',')]
    for layer_id in num_layers:
        args.layer_id = layer_id
        # print(f"Processing layer {layer_id}...")
        args.output_folder_ = os.path.join(args.output_folder, f"layer_{layer_id}")
        collect(positive_file=args.caption_file, negative_file=negative_file, processor=processor, tokenizer=tokenizer, model=model, vanilla_model=vanilla_model, args=args)

if __name__ == "__main__":
    main()