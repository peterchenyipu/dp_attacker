python eval_generic.py --config-name=attack_config 'attack=patch_can_ph_cnn'  ckpt=can_ph_cnn_best log_wandb=true
python eval_generic.py --config-name=attack_config 'attack=patch_can_ph_tf'  ckpt=can_ph_tf_best log_wandb=true
python eval_generic.py --config-name=attack_config 'attack=patch_lift_ph_cnn'  ckpt=lift_ph_cnn_best log_wandb=true
python eval_generic.py --config-name=attack_config 'attack=patch_lift_ph_tf'  ckpt=lift_ph_tf_best log_wandb=true
python eval_generic.py --config-name=attack_config 'attack=patch_square_ph_cnn'  ckpt=square_ph_cnn_best log_wandb=true
python eval_generic.py --config-name=attack_config 'attack=patch_square_ph_tf'  ckpt=square_ph_tf_best log_wandb=true
python eval_generic.py --config-name=attack_config 'attack=patch_toolhang_ph_cnn'  ckpt=toolhang_ph_cnn_best log_wandb=true
python eval_generic.py --config-name=attack_config 'attack=patch_toolhang_ph_tf'  ckpt=toolhang_ph_tf_best log_wandb=true