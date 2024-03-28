<p align="center">
  <img src="mebooster.png" width="80" alt="Logo">
</p>

# **Unlocking High-Fidelity Learning: Towards Neuron-Grained Model Extraction**

Welcome to the official repository of "Unlocking High-Fidelity Learning: Towards Neuron-Grained Model Extraction". This work is dedicated to advancing the field of model extraction by shifting the focus from selecting optimal query samples to enhancing the training process itself. Our novel contribution, MEBooster, is a versatile training booster framework designed to seamlessly integrate with all existing model extraction techniques.

## **Overview**

MEBooster introduces a two-phased approach: an initial bootstrapping phase followed by a constrained gradient backpropagation phase in post-processing. This methodology has demonstrated a significant fidelity gain of up to 58.10%.

## **Supported Model Extraction Attacks**

MEBooster enhances three cutting-edge model extraction attacks:

- **[ActiveThief](https://github.com/gopalaniyengar/activethief):** A pool-based model extraction technique.
- **[DFME](https://github.com/cake-lab/datafree-model-extraction):** A data-free model extraction approach.
- **[MAZE](https://github.com/sanjaykariyappa/MAZE):** A data-free model extraction strategy.
- **[DisGUIDE](https://github.com/lin-tan/disguide):** A data-free model extraction strategy.
## **Getting Started**

### 1. Victim Models

To obtain a victim model, execute the following:

```bash
python mebooster/victim/train.py
```

Configuration for the output path is located in `config.py` under the `mebooster` variable.

### 2. Parameter Estimation

For parameter estimation, run:

```bash
python mebooster/adversarial/moment-based_parameter_estimation.py
```

Parameters for estimation are specified within the script. The output path defaults to `./data_ini/...`.

### 3. Executing DisGUIDE, MAZE and DFME Attacks

To perform MAZE and DFME attacks, including baseline, width expansion, and MEBooster variations without constrained gradient backpropagation, use:

```bash
python mebooster/adversarial/dfme_attacker.py attack_type='DISGUIDE'/'MAZE'/'DFME'
```

Adjust attack parameters in `config.py`, specifying `test_dataset`, `attack_model_arch`, and `victim_model_arch`.

### 4. Post-processing Fine Tuning

After executing MAZE and DFME attacks, apply post-processing fine tuning by running:

```bash
python mebooster/adversarial/dfme_attacker(post-processing).py
```

This script facilitates experiments with MEBooster and regular fine-tuning.

For pool-based attacks, execute:

```bash
python mebooster/adversarial/pool_based_attacker.py
```

This will perform baseline, width expansion, and MEBooster variations without fine-tuning. Results are stored in `models/adversary/ADV_DIR/f{cfg.test_dataset}`, and generated data are saved under `./dfme_data/data_dfme`.

