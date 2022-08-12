====
1, Get victim models

Run mebooster/victim/train.py to get a victim model.
The output path is set in config.py: VICTIM_DIR.

====
2, Parameter esitmation
Run mebooster/adversarial/moment-based_parameter_estimation.py
the params for estimation is set in the file.
The output path is './data_ini/...', which is set in the file.

===
3, Try to Run MAZE and DFME attacks
Run mebooster/adversarial/dfme_attacker.py
It can perform the experiments (Baseline; Stratgy-o; Strategy-oi).
For any attacks, set the params in config.py: test_dataset, attack_model_arch, victim_model_arch.

===
4, Try to run layer-wise training after MAZE and DFME attacks
Run mebooster/adversarial/dfme_attacker(layer-wise training).py
It can perform the experiments (Stratgy-oil; Strategy-oip).

===
5, Try to Run Pool-based attacks
Run mebooster/adversarial/pool_based_attacker.py
It can perform the experiments (Baseline; Stratgy-o; Strategy-oi).
The model/model extraction results are in 'models/adversary/ADV_DIR/f{cfg.test_dataset}';
The data generated are in './dfme_data/data_dfme'
===
6, Try to run layer-wise training after pool-based attacks
Run mebooster/adversarial/pool_based_attacker(layer-wise training).py
It can perform the experiments (Stratgy-oil; Strategy-oip).

===
7, Run Transfer learning related experiments
Run the .py files with suffix (transfer_learn).