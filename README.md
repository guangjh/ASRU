# ASRU: Activation Steering Meets Reinforcement Unlearning for Multimodal Large Language Models
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white&labelColor=555555">
</p>

---

## 📖 Introduction

Multimodal large language models (MLLMs) may memorize sensitive cross-modal information during pretraining, making machine unlearning (MU) crucial. Existing methods typically evaluate unlearning effectiveness based on output deviations, while overlooking the generation quality after unlearning. This can easily lead to hallucinated or rigid responses, thereby affecting the usability and safety of the unlearned model. To address this issue, we propose ASRU, a controllable multimodal unlearning framework that incorporates generation quality as a core evaluation objective. ASRU first induces initial refusal behavior through activation redirection, and then optimizes fine-grained refusal boundaries using a customized reward function, thereby achieving a better trade-off between target knowledge unlearning and model utility.

![framework-asru](https://github.com/guangjh/ASRU/blob/main/assets/framework-asru.png)

## 📢 News
- **[2026.05.01]** Our paper is accepted by ICML 2026.


## ⚙️ Install

```txt
python==3.10
torch==2.8.0
transformers==4.57.0
vllm==0.11.0
flash-attn==2.8.3
```

## 📥 Download Dataset

Please download the required datasets from the following links:
- **MLLMU-Bench**: [Train Set](https://huggingface.co/MLLMMU/baseline_train_split) | [Test Set](https://huggingface.co/datasets/MLLMMU/MLLMU-Bench)

## 🤖 Getting Vanilla Model

To obtain the vanilla models used in our experiments, please run the following commands.

For **MLLMU-Bench**:

```bash
python MLLMU_finetune.py \
  --model_id path_to_original_model \
  --forget_split_ratio 5 \
  --save_dir path_to_mllmu_vanilla_model
```

## 🚀 Running Baselines

Detailed instructions for training baseline unlearning methods are available [here](your_link).

## 🚀 Running ASRU

### Getting the Steering Vector VL*

To construct the steering vector VL*, first download 400 unseen images from [DigiFace-1M](https://github.com/microsoft/DigiFace1M).
These images are used to extract the reference activation direction.

To collect activations for obtaining the steering vector VL*, run:

```bash
python collect_activations.py \
  --model_id "{path_to_original_model}" \
  --cache_path "{path_to_vanilla_model}" \
  --data_split_folder data_split" \
  --forget_ratio "{forget_ratio}" \
  --output_folder "{output_activation_folder}" \
  --caption_image_folder "{unseen_images_path}" \
  --caption_data_size 400 \
  --layer_ids 17 \
  --negative_file_type "forget"
```
`caption_image_folder` denotes the folder containing the 400 unseen images downloaded from DigiFace-1M.
### Activation Steering Optimization

After obtaining the steering vector VL*, train ASRU by running:

```bash
layer_id=17
forget_ratio=5

python act_steering.py \
  --model_id "{path_to_original_model}" \
  --vanilla_dir "{path_to_vanilla_model}" \
  --data_split_dir "{data_split_dir}" \
  --forget_split_ratio "${forget_ratio}" \
  --output_dir "{output_dir}" \
  --layer_id "${layer_id}" \
  --layer_ids "${layer_id}" \
  --batch_size 2 \
  --lr 1e-4 \
  --forget_scale 3 \
  --alpha 1 \
  --num_epochs 3 \
  --activation_dir "{activation_root}/layer_${layer_id}"
```
### Refusal Boundary Optimization via GRPO

Download the boundary set from [ASRU-data](https://huggingface.co/datasets/closerG/ASRU-data).

To construct the boundary set locally, run:

```bash
python data_process/process.py
```
After obtaining the activation-steered model, run reinforcement unlearning with:

```bash
bash qwen_mllmu_grpo.sh
```
Then merge the model
```
cd EasyR1
python scripts/model_merger.py --local_dir path_to_model
```
## 🚀 Evaluation

To evaluate the models, run the following commands:

```
cd eval
bash MLLMU_eval.sh
```
To evaluate generation quality with GPT-4o-mini, run:
```
python eval/natural_evaluation.py \
  --json_path /path/to/your_result.json \
  --output_dir /path/to/output_dir

```
## 🙏 Acknowledgement

We are highly inspired by [EasyR1](https://github.com/hiyouga/EasyR1).

