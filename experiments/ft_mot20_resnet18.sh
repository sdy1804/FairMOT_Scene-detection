cd src
python train.py --task mot --exp_id ft_mot20_resnet18 --data_cfg '../src/lib/cfg/SWIM.json' --num_epochs 10 --lr_step '50' --batch_size 1
cd ..