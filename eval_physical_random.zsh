python eval_generic.py --config-name=attack_config 'attack=patch_random'  ckpt=can_ph_cnn_best log_wandb=true
python eval_generic.py --config-name=attack_config 'attack=patch_random'  ckpt=can_ph_tf_best log_wandb=true
python eval_generic.py --config-name=attack_config 'attack=patch_random'  ckpt=lift_ph_cnn_best log_wandb=true
python eval_generic.py --config-name=attack_config 'attack=patch_random'  ckpt=lift_ph_tf_best log_wandb=true
python eval_generic.py --config-name=attack_config 'attack=patch_random'  ckpt=square_ph_cnn_best log_wandb=true
python eval_generic.py --config-name=attack_config 'attack=patch_random'  ckpt=square_ph_tf_best log_wandb=true
python eval_generic.py --config-name=attack_config 'attack=patch_random'  ckpt=toolhang_ph_cnn_best log_wandb=true
python eval_generic.py --config-name=attack_config 'attack=patch_random'  ckpt=toolhang_ph_tf_best log_wandb=true