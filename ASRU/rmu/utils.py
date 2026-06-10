import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
random.seed(0)
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import LlavaForConditionalGeneration, AutoProcessor, get_scheduler, MllamaForConditionalGeneration, AutoTokenizer,Qwen2VLForConditionalGeneration,BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration



################################
##### Activation functions #####
################################

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
    
    # return cache[0].mean(1)  # mean pool over sequence length
    return cache[0][:, -1, :]
    
#######################################
##### Model and data loading code #####
#######################################

def find_all_linear_names(model):
    print(model)
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["embeddings","embed_tokens","patch_embed"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else ".".join(names[-2:]))

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    # if "qwen" in str(model).lower():
    #     lora_module_names.remove('proj')
    return list(lora_module_names)

def get_params(model, layer_ids, param_ids=None):
    params = []
    for layer_id in layer_ids:
        for i, p in enumerate(model.language_model.model.layers[layer_id].parameters()):
            if param_ids:
                if i in param_ids:
                    params.append(p)
            else:
                params.append(p)
    return params


def get_params_mlp(args, model, module_list=['language_model'], layer_ids=None):
    params = []
    if 'language_model' in module_list:
        if layer_ids:
            print("Getting MLP params for layers in language model: ", layer_ids)
            for layer_id in layer_ids:
                # 获取指定层
                if "qwen" in args.model_id.lower():
                    layer = model.language_model.layers[layer_id]
                else:
                    layer = model.language_model.model.layers[layer_id]

                
                # 检查 MLP 是否存在
                if hasattr(layer, "mlp"):
                    mlp = layer.mlp
                else:
                    raise ValueError(f"Layer {layer_id} does not have an MLP module.")

                # #提取 MLP 中的线性层权重
                # linear_layers = [mlp.gate_proj, mlp.up_proj, mlp.down_proj]
                # for i, linear_layer in enumerate(linear_layers):
                #     params.append(linear_layer.weight)  # 默认提取所有线性层的权重
                params.append(mlp.down_proj.weight)
        else:
            print("Getting MLP params for all layers")
            layers = model.language_model.model.layers
            for layer in layers:
                # 检查 MLP 是否存在
                if hasattr(layer, "mlp"):
                    mlp = layer.mlp
                else:
                    raise ValueError(f"Layer {layer} does not have an MLP module.")
                params.append(mlp.down_proj.weight)
    if 'vision_tower' in module_list:
        print("Getting MLP params for vision_tower")
        layers = model.vision_tower.vision_model.encoder.layers
        for layer in layers:
            # 检查 MLP 是否存在
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
            else:
                raise ValueError(f"Layer does not have an MLP module.")
            params.append(mlp.fc2.weight)
    if 'multi_modal_projector' in module_list:
        print("Getting MLP params for multi_modal_projector")
        mlp = model.multi_modal_projector
        params.append(mlp.linear_2.weight)
    return params


def load_model(model_name_or_path):
    torch_dtype = "auto" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer

def load_model_and_processor(args, lora_model=False, original=False):
    """
    Load the model and processor based on the provided model_id.
    Different models may require different loading methods, which are handled with conditional statements.
    """
    if "llava" in args.model_id:
        # Load LLAVA model and processor
        print("Loading LLAVA model...")
        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.tokenizer.padding_side = "right"  # Ensure right padding
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
        if original:
            model = LlavaForConditionalGeneration.from_pretrained(
                args.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
        else:
            model = LlavaForConditionalGeneration.from_pretrained(
                args.vanilla_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
        if not lora_model:
            return model, processor
        print("getting peft model")
        lora_config = LoraConfig(
            r=16, #32
            lora_alpha=16, #8
            lora_dropout=0.05,
            # target_modules=["q_proj", "v_proj"],
            target_modules=["language_model.model.layers.17.mlp.down_proj"],
            init_lora_weights="gaussian",
        )
        # model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model, processor

    elif "llama" in args.model_id.lower():
        model = MllamaForConditionalGeneration.from_pretrained(
            args.vanilla_dir, 
            device_map="auto",
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            local_files_only=True,
        )

        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.tokenizer.padding_side = "right"  # Ensure right padding
    elif "qwen" in args.model_id.lower():
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.vanilla_dir, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.tokenizer.padding_side = "right"  # Ensure right padding
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    return model, processor

def get_data(forget_corpora, retain_corpora, min_len=50, max_len=2000, batch_size=4):
    def get_dataset(name):
        data = []
        if name == "wikitext":
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for x in raw_data:
                if len(x['text']) > min_len:
                    data.append(str(x['text']))
        else:
            assert os.getenv("HF_TOKEN"), "HF_TOKEN is not set"
            dataset = load_dataset(f"cais/wmdp-{name}", split="train", token=os.getenv("HF_TOKEN"))
            for line in dataset:
                if len(line['text']) > min_len:
                    data.append(str(line['text']))
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return data

    return (
        [get_dataset(c) for c in forget_corpora],
        [get_dataset(c) for c in retain_corpora]
    )


def get_mllmu_data(data_split_dir, forget_split_ratio, min_len=50, max_len=2000, batch_size=4):
    forget_folder = os.path.join(data_split_dir, f"forget_{forget_split_ratio}")
    retain_folder = os.path.join(data_split_dir, f"retain_{100 - forget_split_ratio}")
    print("Forget Folder: ", forget_folder)
    print("Retain Folder: ", retain_folder)

    # Define paths to the Parquet files for "forget" and "retain" datasets
    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")

    # Load DataLoader
    forget_df = pd.read_parquet(forget_parquet_file)
    retain_df = pd.read_parquet(retain_parquet_file)
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
        flattened_data = [flattened_data[i:i + batch_size] for i in range(0, len(flattened_data), batch_size)]
        return flattened_data
    return (
        [get_dataset(forget_df)],
        [get_dataset(retain_df)]
    )