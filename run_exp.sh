
datasize=500
# zero shot
python3 run_gsm8k.py --data_size $datasize --few_shot_num 0
# few shot
# semantic, w/o label
python3 run_gsm8k.py --data_size $datasize --few_shot_num 1 --strategy semantic --unsupervised
# random, w/o label
python3 run_gsm8k.py --data_size $datasize --few_shot_num 1 --strategy random --unsupervised
# semantic, w label
python3 run_gsm8k.py --data_size $datasize --few_shot_num 1 --strategy semantic
# random, w/o label
python3 run_gsm8k.py --data_size $datasize --few_shot_num 1 --strategy random