
cd /shared/deep-learning-models/models/nlp/
export PYTHONPATH=${PYTHONPATH}:${PWD}

smddprun -n 16 \
-x LD_LIBRARY_PATH \
-x PATH \
-x FI_PROVIDER="efa" \
-x NCCL_DEBUG=INFO \
-x PYTHONPATH \
python /shared/deep-learning-models/models/nlp/albert/run_pretraining.py \
--train_dir=/scratch/data/train/max_seq_len_128_max_predictions_per_seq_20_masked_lm_prob_15 \
--val_dir=/scratch/data/validation/max_seq_len_512_max_predictions_per_seq_20_masked_lm_prob_15 \
--log_dir=/shared/logs \
--checkpoint_dir=/shared/checkpoints \
--load_from=scratch \
--model_type=albert \
--model_size=large \
--per_gpu_batch_size=12 \
--gradient_accumulation_steps=1 \
--warmup_steps=3125 \
--total_steps=1000 \
--learning_rate=0.00176 \
--optimizer=lamb \
--log_frequency=10

