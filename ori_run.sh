torchrun --nproc_per_node 1 ori_example_chat_completion.py \
    --ckpt_dir /root/llama2-7b-chat/ \
    --tokenizer_path /root/llama2-7b-chat/tokenizer.model \
    --max_seq_len 3500 --max_batch_size 8 --input_len 2048