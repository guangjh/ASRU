import sys
import os
import datetime
sys.path.append(('.'))
sys.path.append(('../'))
sys.path.append(('../../'))

import numpy as np
import torch
from transformers import get_scheduler
from tqdm import tqdm
from peft import PeftModel
import pandas as pd
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from accelerate import Accelerator

from rmu.utils import load_model, get_params, get_params_mlp, forward_with_cache, get_data, get_mllmu_data, load_model_and_processor

from data_process.data_preprocess import LLAVA_multimodal_Dataset, train_collate_fn_mllmu, train_collate_mllmu_ansonly,Vanilla_LLaVA_Dataset
from data_process.CLEAR_process import CLEAR_Dataset, CAPTION_MODE, RECOGNITION_MODE, train_collate_clear, NONE_MODE,train_collate_clear_ansonly
from itertools import cycle
from qwen_vl_utils import process_vision_info

def run_rmu_mllmu(
    updated_model,
    frozen_model,
    tokenizer,
    args,
):
    updated_model = updated_model.train()

    # params = get_params(updated_model, args.layer_ids, args.param_ids)
    params = get_params_mlp(args, updated_model, module_list=['language_model'], layer_ids=args.layer_ids)

    optimizer = AdamW(params, lr=args.lr)
    frozen_module = eval(
        args.module_str.format(model_name="frozen_model", layer_id=args.layer_id)
    )
    updated_module = eval(
        args.module_str.format(model_name="updated_model", layer_id=args.layer_id)
    )

    activation_path = os.path.join(args.activation_dir, "activation_difference.pt")
    if not os.path.exists(activation_path):
        raise FileNotFoundError(f"Activation file not found at {activation_path}")
    control_vec = torch.load(activation_path).to(updated_model.device)
    print(f"Loaded control_vec from {activation_path}, shape: {control_vec.shape}")

    # control_vec = control_vec.unsqueeze(0).expand(args.batch_size, -1, -1)
    print(f"Expanded control_vec to match batch_size={args.batch_size}, new shape: {control_vec.shape}")


    tokenizer = processor.tokenizer
    print("Tokenizer Length: ", len(tokenizer))

    # Resize token embeddings to match the tokenizer
    updated_model.resize_token_embeddings(len(processor.tokenizer))
    if len(tokenizer) > updated_model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        updated_model.resize_token_embeddings(len(tokenizer))
    frozen_model.resize_token_embeddings(len(processor.tokenizer))
    if len(tokenizer) > frozen_model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        frozen_model.resize_token_embeddings(len(tokenizer))
    
    if isinstance(updated_model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

    # Dataset and Dataloader setup

    forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_split_ratio}")
    retain_folder = os.path.join(args.data_split_dir, f"retain_{100 - args.forget_split_ratio}")
    print("Forget Folder: ", forget_folder)
    print("Retain Folder: ", retain_folder)

    # Define paths to the Parquet files for "forget" and "retain" datasets
    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")

    # Load DataLoader
    forget_df = pd.read_parquet(forget_parquet_file)
    retain_df = pd.read_parquet(retain_parquet_file)


    multimodal_forget_dataset = LLAVA_multimodal_Dataset(df=forget_df)
    multimodal_retain_dataset = LLAVA_multimodal_Dataset(df=retain_df)


    if processor:
        train_dataloader_forget = DataLoader(
            multimodal_forget_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_clear(x, processor, updated_model.device,True)
        )

        train_dataloader_retain = DataLoader(
            multimodal_retain_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_clear(x, processor, updated_model.device,True)
        )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    if args.grad_mask_path:
        module_set=set()
        grad_data=torch.load(args.grad_mask_path)
        grad_mask=grad_data['weight']
        layer_name_list=list(grad_mask.keys())
        for name in layer_name_list:
            if "proj" in name:
                continue
            elif "fc" in name:
                continue
            elif "linear" in name:
                continue
            elif "mlp" in name:
                continue
            elif "qkv" in name:
                continue
            else:
                grad_mask.pop(name)
        def calc_sparsity(tensor):
            num_zero_elements = tensor.numel() - torch.count_nonzero(tensor)
            total_elements = tensor.numel()
            sparsity = num_zero_elements / total_elements
            return sparsity.item(), total_elements, num_zero_elements

        total_cnt=0
        w_cnt=0
        for k,v in grad_mask.items():
            try: 
                w_sparsity, total_elements, w_num_zero_elements = calc_sparsity(v)
                total_cnt += total_elements
                w_cnt += w_num_zero_elements
                module_set=module_set|set(k.split("."))
            except: 
                pass 
        print("Saliency mask generated!")
        #print(f"Total sparsity among weight:{w_cnt/total_cnt*100}")

        print(f"Fine-tuned modules: {module_set}\n",)


    accelerator = Accelerator()

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader_forget) + len(train_dataloader_retain)) * args.num_epochs,
    )
    # ==== Prepare model/data/optimizer ====
    updated_model, optimizer, train_dataloader_forget, train_dataloader_retain, lr_scheduler = accelerator.prepare(
        updated_model, optimizer, train_dataloader_forget, train_dataloader_retain, lr_scheduler
    )

    len_retain, len_forget = len(train_dataloader_retain), len(train_dataloader_forget)
    n_iters = len(train_dataloader_retain)
    forget_freq = len(train_dataloader_retain) // len(train_dataloader_forget)
    print(f"Each epoch have {n_iters} iterations, with forget_freq={forget_freq}!")
    for epoch in range(args.num_epochs):
        train_data_forget = enumerate(train_dataloader_forget)
        train_data_retain = enumerate(train_dataloader_retain)
        total_loss_forget = 0.0
        total_loss_retain = 0.0

        for iter in tqdm(range(0, n_iters)):
            if iter % forget_freq == 0:  # update only in specific forget round
                try:
                    _, forget_batch = next(train_data_forget)

                    updated_forget_activations = forward_with_cache(
                        updated_model, forget_batch, module=updated_module, no_grad=False
                    ).to(updated_model.device)
                    frozen_forget_activations = forward_with_cache(
                        frozen_model, forget_batch, module=frozen_module, no_grad=True
                    ).to(updated_model.device)

                    # ====== 统一 shape：都变成 [B, H] ======
                    def to_batch_hidden(x):
                        # 常见三种情况： [B, T, H] / [B, 1, H] / [B, H]
                        if x.dim() == 3:
                            if x.size(1) == 1:
                                return x[:, 0, :]        # [B, H]
                            else:
                                return x[:, -1, :]       # 取最后一个 token
                        elif x.dim() == 2:
                            return x                     # 已经是 [B, H]
                        else:
                            raise ValueError(f"Unexpected activation shape: {x.shape}")

                    updated_forget_last = to_batch_hidden(updated_forget_activations)   # [B, H]
                    frozen_forget_last  = to_batch_hidden(frozen_forget_activations)    # [B, H]

                    # control_vec: [H] 或 [1, H]，扩成 [1, H] 让它 broadcast 到 [B, H]
                    control_vec_ = control_vec.to(updated_model.device)
                    if control_vec_.dim() == 1:
                        control_vec_ = control_vec_.unsqueeze(0)     # [1, H]
                    elif control_vec_.dim() == 2:
                        pass                                        # [1, H] or [B, H]
                    else:
                        raise ValueError(f"Unexpected control_vec shape: {control_vec_.shape}")

                    target = frozen_forget_last + args.forget_scale * control_vec_  # [B, H]（或 [1,H] 自动广播到 [B,H]）

                    # 最重要的一步：保证两边 shape 一致
                    assert updated_forget_last.shape == target.shape, \
                        f"updated {updated_forget_last.shape}, target {target.shape}"

                    loss_forget = torch.nn.functional.mse_loss(
                        updated_forget_last, target
                    )

                    accelerator.backward(loss_forget)
                    # accelerator.clip_grad_norm_(updated_model.parameters(), max_norm=1.0)

                    # if args.grad_mask_path:
                    #     for name, p in updated_model.named_parameters():
                    #         if p.grad is not None and name in grad_mask:
                    #             # print(f"Grad of {name} masked")
                    #             p.grad *= grad_mask[name].to(p.grad.device)
                    #             # print(p.grad)
                    #     # exit()
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss_forget += loss_forget.item()

                except Exception as e:
                    print("Exception occurred while fetching forget batch:", e)
                    pass

            try:
                _, retain_batch = next(train_data_retain)
            except StopIteration:
                print("Reached the end of train_data_retain iterator. Skipping this iteration.")
                continue

            updated_retain_activations = forward_with_cache(
                updated_model, retain_batch, module=updated_module, no_grad=False
            ).to(updated_model.device)
            frozen_retain_activations = forward_with_cache(
                frozen_model, retain_batch, module=frozen_module, no_grad=True
            ).to(updated_model.device)

            loss_retain_raw = torch.nn.functional.mse_loss(
                updated_retain_activations, frozen_retain_activations
            )
            loss_retain = args.alpha[0] * loss_retain_raw

            accelerator.backward(loss_retain)
            # accelerator.clip_grad_norm_(updated_model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss_retain += loss_retain.item()

            print(f"Iteration {iter + 1} - Forget Loss: {loss_forget.item():.4f}")
            print(f"Iteration {iter + 1} - Retain Loss: {loss_retain.item():.4f}")
        lr_scheduler.step()
        print(f"Epoch {epoch + 1} - Forget Loss: {total_loss_forget / len_forget}")
        print(f"Epoch {epoch + 1} - Retain Loss: {total_loss_retain / len_retain}")

        # # Save the model at the end of each epoch
        # model_save_path = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
        # os.makedirs(model_save_path, exist_ok=True)
        # accelerator.wait_for_everyone()
        # unwrapped_model = accelerator.unwrap_model(updated_model)
        # if isinstance(updated_model, PeftModel):
        #     unwrapped_model = unwrapped_model.merge_and_unload()
        # unwrapped_model.save_pretrained(model_save_path)
        # print(f"Model for epoch {epoch + 1} saved to: {model_save_path}")

    # Save the final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(updated_model)
    if isinstance(updated_model, PeftModel):
        unwrapped_model = unwrapped_model.merge_and_unload()
    unwrapped_model.save_pretrained(args.output_dir)
    print(f"Model saved to: {args.output_dir}")


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    ### Model arguments
    parser.add_argument(
        "--model_id", type=str, default="HuggingFaceH4/zephyr-7b-beta"
    )
    parser.add_argument(
        "--vanilla_dir", type=str, default="HuggingFaceH4/zephyr-7b-beta"
    )
    parser.add_argument(
        "--module_str", type=str, default="{model_name}.language_model.layers[{layer_id}]"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None
    )

    parser.add_argument("--data_split_dir", type=str, default="../Data_split", help="Directory to save the model")
    parser.add_argument("--forget_split_ratio", type=int, default=5, help="Directory to save the model")

    parser.add_argument("--alpha", type=str, default="1", help="retain weight")

    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--forget_scale", type=float, default=1.0)

    parser.add_argument("--min_len", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_num_batches", type=int, default=80)
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--layer_id", type=int, default=5, help="layer to unlearn")
    parser.add_argument("--layer_ids", type=str, default="5", help="update layers")
    parser.add_argument("--param_ids", type=str, default="7", help="update params")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--verbose", action="store_true", help="Logging the activations norms and cosine at each step")
    parser.add_argument("--grad_mask_path", type=str, default=None, help="Mask for gradient update")

    parser.add_argument("--activation_dir", type=str, required=True, help="Directory where activation differences are stored")

    args = parser.parse_args()
    args.alpha = [float(c) for c in args.alpha.split(",")]
    args.layer_ids = [int(layer_id) for layer_id in args.layer_ids.split(",")]
    args.param_ids = [int(param_id) for param_id in args.param_ids.split(",")]
    return args 


if __name__ == "__main__":
    
    args = get_args()

    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if "qwen" in args.model_id.lower():
        from torch.optim import AdamW
    else:
        from transformers import AdamW

    frozen_model, processor = load_model_and_processor(args, lora_model=False)
    updated_model, processor = load_model_and_processor(args, lora_model=False)

    run_rmu_mllmu(
        updated_model,
        frozen_model,
        processor,
        args,
    )
