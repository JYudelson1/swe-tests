set -x
source .env

uv lock --upgrade-package openrlhf

uv run ray stop > /dev/null 2>&1 #Suppress the output of ray stop

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run ray start --head --port 6380 --num-gpus 8 --num-cpus 128 > /dev/null 2>&1

uv run ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"], "env_vars": {}}' \
  --working-dir . \
  -- python -m openrlhf.cli.train_ppo_ray \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 2 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 4 \
  --vllm_num_engines 2 \
  --vllm_tensor_parallel_size 1 \
  --use_kl_loss \
  --use_kl_estimator_k3 \
  --pretrain deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --save_path checkpoint/bash_testing \
  --micro_train_batch_size 1 \
  --train_batch_size 4 \
  --micro_rollout_batch_size 2 \
  --rollout_batch_size 4 \
  --n_samples_per_prompt 4 \
  --max_samples 300 \
  --max_epochs 1 \
  --prompt_max_len 10000 \
  --generate_max_len 5000 \
  --zero_stage 3 \
  --bf16 \
  --actor_learning_rate 1e-7 \
  --init_kl_coef 1e-3 \
  --prompt_data bashbenchsimple.json \
  --input_key problem_statement \
  --apply_chat_template \
  --normalize_reward \
  --packing_samples \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb b936d9b7656895a60f4f4f993e3b1a7639b3430c \
  --wandb_project bash_bench_rl \
  --advantage_estimator grpo \
  --env_file bash_bench_env \
  --env_class BashBenchEnv \
  --eval_steps 1 \
  --adam_offload \
  --enforce_eager \
  --wandb_run_name dense_and_truncated \
  # --lora_rank 64 \
  # --lora_alpha 128 \
  # --use_kl_estimator_k3 \

  # --ref_num_nodes 1 \
  # --ref_num_gpus_per_node 6 \
  # --actor_num_nodes 1 \
  # --actor_num_gpus_per_node 6 \
  # --vllm_num_engines 2 \
  # --vllm_tensor_parallel_size 3 \
  # --colocate_all_models \
  # --vllm_gpu_memory_utilization 0.6 \
  # --vllm_enable_sleep \
  #--ring_head_stride 2 \

# try: train_batch_size = 12, rollout_batch_size = 8, micro_train_batch_size = 1

# the following relationship should be verified:
# micro_train_batch_size * gradient_accumulation_steps * actor num_nodes == train_batch_size
# rollout_batch_size % vllm_num_engines == 0
