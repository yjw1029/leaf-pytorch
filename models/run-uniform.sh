
dataset=$1

if [ $dataset == 'femnist' ]
then
    clients_per_round=35
    num_epochs=1
    num_rounds="2000"
    fedavg_lr="0.004"
    CUDA_VISIBLE_DEVICES=4 python main.py -dataset 'femnist' -model 'cnn' --num-rounds ${num_rounds} --clients-per-round ${clients_per_round} --num-epochs ${num_epochs} -lr ${fedavg_lr} \
    --wandb-run uniform-${clients_per_round}-${num_epochs} --agg-fn uniform

elif [ $dataset == 'sent140' ]
then
    pushd sent140
        if [ ! -f glove.6B.300d.txt ]; then
            ./get_embs.sh
        fi
    popd
    CUDA_VISIBLE_DEVICES=4 python main.py -dataset 'sent140' -model 'stacked_lstm' --num-rounds 10 \
    --clients-per-round 10 --agg-fn uniform --wandb-run uniform
elif [ $dataset == 'celeba' ]
then
    clients_per_round=10
    num_epochs=1
    num_rounds="1000"
    fedavg_lr="0.001"
    CUDA_VISIBLE_DEVICES=0 python main.py -dataset 'celeba' -model 'cnn' --num-rounds ${num_rounds} --clients-per-round ${clients_per_round} --num-epochs ${num_epochs} -lr ${fedavg_lr} \
    --wandb-run uniform --agg-fn uniform
elif [ $dataset == 'shakespeare' ]
then
    num_epochs=1
    CUDA_VISIBLE_DEVICES=0 python -u main.py -dataset shakespeare -model stacked_lstm --seed 0 --num-rounds 80 \
    --clients-per-round 10 --num-epochs ${num_epochs} -lr 0.8 --wandb-run uniform --agg-fn uniform
elif [ $dataset == "reddit" ]
then
    CUDA_VISIBLE_DEVICES=3 python3 main.py -dataset reddit -model stacked_lstm --eval-every 10 --num-rounds 1000 --clients-per-round 10 \
    --batch-size 5 -lr 5.65 --metrics-name reddit_experiment --wandb-run uniform --agg-fn uniform
fi
