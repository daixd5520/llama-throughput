torchrun --nproc_per_node 8 example_chat_completion.py \
    --ckpt_dir /root/Llama2-70b-chat-hf \
    --tokenizer_path /root/Llama2-70b-chat-hf/tokenizer.model \
    --max_seq_len 4096 --max_batch_size 1
