python train_physical_attack.py --config-name=can_ph_tf_best attack_method=gd_lr1 'name=${now:%Y.%m.%d-%H.%M.%S}_${task_name}_${dataset_type}_${arch}_gd'
python train_physical_attack.py --config-name=can_ph_cnn_best attack_method=gd_lr1  'name=${now:%Y.%m.%d-%H.%M.%S}_${task_name}_${dataset_type}_${arch}_gd'
python train_physical_attack.py --config-name=lift_ph_tf_best attack_method=gd_lr1  'name=${now:%Y.%m.%d-%H.%M.%S}_${task_name}_${dataset_type}_${arch}_gd'
python train_physical_attack.py --config-name=lift_ph_cnn_best attack_method=gd_lr1  'name=${now:%Y.%m.%d-%H.%M.%S}_${task_name}_${dataset_type}_${arch}_gd'
python train_physical_attack.py --config-name=square_ph_tf_best attack_method=gd_lr1  'name=${now:%Y.%m.%d-%H.%M.%S}_${task_name}_${dataset_type}_${arch}_gd'
python train_physical_attack.py --config-name=square_ph_cnn_best attack_method=gd_lr1  'name=${now:%Y.%m.%d-%H.%M.%S}_${task_name}_${dataset_type}_${arch}_gd'
python train_physical_attack.py --config-name=toolhang_ph_tf_best attack_method=gd_lr1  'name=${now:%Y.%m.%d-%H.%M.%S}_${task_name}_${dataset_type}_${arch}_gd'
python train_physical_attack.py --config-name=toolhang_ph_cnn_best attack_method=gd_lr1  'name=${now:%Y.%m.%d-%H.%M.%S}_${task_name}_${dataset_type}_${arch}_gd'