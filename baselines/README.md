# 🚀 Baselines Training
Here we provide instructions on how to train baselines.
## GA
```bash
python MLLMU_GA.py \
  --model_id "{path_to_original_model}" \
  --vanilla_dir "{vanilla_dir}" \
  --data_split_dir MLLMMU-Bench/data_split_dir \
  --forget_split_ratio "{forget_ratio}" \
  --save_dir "{output_dir}" \
  --batch_size 4 \
  --lr 2e-5 \
  --num_epochs 4 \
  --ans_only True
```
## GA_diff
```bash
python MLLMU_GA_Diff.py \
  --model_id "{path_to_original_model}" \
  --vanilla_dir "{vanilla_dir}" \
  --data_split_dir MLLMMU-Bench/data_split_dir \
  --forget_split_ratio "{forget_ratio}" \
  --save_dir "{output_dir}" \
  --batch_size 4 \
  --lr 2e-5 \
  --num_epochs 4 \
  --ans_only True
```
## KL_min
```bash
python MLLMU_KL_Min.py \
  --model_id "{path_to_original_model}" \
  --vanilla_dir "{vanilla_dir}" \
  --data_split_dir MLLMMU-Bench/data_split_dir \
  --forget_split_ratio "{forget_ratio}" \
  --save_dir "{output_dir}" \
  --batch_size 4 \
  --lr 2e-5 \
  --num_epochs 4 \
  --ans_only True
```
## NPO
```bash
python MLLMU_NPO.py \
  --model_id "{path_to_original_model}" \
  --vanilla_dir "{vanilla_dir}" \
  --oracle_model_id "{vanilla_dir}" \
  --data_split_dir "{data_split_dir}" \
  --forget_split_ratio "{forget_ratio}" \
  --save_dir "{output_dir}" \
  --batch_size 4 \
  --lr 1e-5 \
  --num_epochs 4 \
  --ans_only True
```
## MMunlearner
To generate the gradient mask, run:
```
cd data_process
python data_process/MLLMU_gen_mask.py
```
To run **MMUnlearner** on **MLLMU-Bench**, use:

```bash
python MLLMU_manifold.py \
  --model_id "{path_to_original_model}" \
  --vanilla_dir "{vanilla_dir}" \
  --data_split_dir "{data_split_dir}" \
  --forget_split_ratio "{forget_ratio}" \
  --save_dir "{output_dir}" \
  --batch_size 4 \
  --lr 2e-5 \
  --num_epochs 4 \
  --grad_mask_path "{path_to_grad_mask}" \
  --ans_only True
```
