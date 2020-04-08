for n_classes in 2 3
do
  python tasks/hss.py --window-size 0.08 --window-stride 0.06 --batch-size 64 --model-type cnn \
  --cuda --n-jobs 1 --expt-id debug --epoch-rate 1.0 --epochs 20 --tensorboard --sample-balance same \
  --amp --test --n-mels 64 --return-prob --loss-func ce --n-classes $n_classes

  for model in rnn cnn_rnn
  do
    for rnn_type in lstm gru
    do
      python tasks/hss.py --window-size 0.08 --window-stride 0.06 --batch-size 64 --model-type $model \
      --cuda --n-jobs 1 --expt-id debug --epoch-rate 1.0 --epochs 20 --tensorboard --sample-balance same \
      --amp --test --n-mels 64 --return-prob --loss-func ce --n-classes $n_classes --rnn-type $rnn_type
    done
  done
done