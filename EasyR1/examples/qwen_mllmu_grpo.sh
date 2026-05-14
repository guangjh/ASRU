#!/bin/bash

set -x

unset RAY_ADDRESS
unset RAY_HEAD_IP
unset RAY_GCS_ADDRESS
unset RAY_REDIS_ADDRESS

MODEL_PATH="{path_to_activation_steered_model}"

CUDA_VISIBLE_DEVICES=0,1,2,3 stdbuf -oL -eL python3 -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files="{path_to_train_file}" \
  data.val_files="{path_to_val_file}" \
  worker.actor.model.model_path="${MODEL_PATH}" \
  trainer.experiment_name="asru_grpo" \
  trainer.n_gpus_per_node=4 \
  algorithm.kl_coef=0.1 \
  worker.reward.reward_function="./examples/reward_function/all_reward_copy.py:compute_score" \
  worker.val_reward.reward_function="./examples/reward_function/all_reward_copy.py:compute_score" \
  trainer.total_epochs=6 \
  trainer.save_freq=5 \
  trainer.val_freq=5