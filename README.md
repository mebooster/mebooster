<p align="center">
  <img src="mebooster.png" width="80" alt="Logo">
</p>

# **MEBooster: A General Booster Framework for Learning-Based Model Extraction**

Welcome to the official repository of "MEBooster: A General Booster Framework for Learning-Based Model Extraction." This work is dedicated to advancing the field of model extraction by shifting the focus from selecting optimal query samples to enhancing the training process itself. Our novel contribution, MEBooster, is a versatile training booster framework designed to seamlessly integrate with all existing model extraction techniques.

## **Overview**

MEBooster introduces a two-phased approach: an initial bootstrapping phase followed by a constrained gradient backpropagation phase in post-processing. This methodology has demonstrated a significant fidelity gain of up to 58.10% in image classification tasks. Furthermore, by implementing dual optimizations within the Transformer model, MEBooster achieves remarkable efficiency improvements, including a 26.5x speedup at a sequence length of 512 and an 80% reduction in communication overhead, offering up to a 10x acceleration over current leading private inference frameworks.

## **Supported Model Extraction Attacks**

MEBooster enhances three cutting-edge model extraction attacks:

- **[ActiveThief](https://github.com/gopalaniyengar/activethief):** A pool-based model extraction technique.
- **[DFME](https://github.com/cake-lab/datafree-model-extraction):** A data-free model extraction approach.
- **[MAZE](https://github.com/sanjaykariyappa/MAZE):** Another data-free model extraction strategy.

## **Getting Started**

### 1. Victim Models

To obtain a victim model, execute the following:

```bash
python mebooster/victim/train.py
```

Configuration for the output path is located in `config.py` under the `VICTIM_DIR` variable.

### 2. Parameter Estimation

For parameter estimation, run:

```bash
python mebooster/adversarial/moment-based_parameter_estimation.py
```

Parameters for estimation are specified within the script. The output path defaults to `./data_ini/...`.

### 3. Executing MAZE and DFME Attacks

To perform MAZE and DFME attacks, including baseline, width expansion, and MEBooster variations without constrained gradient backpropagation, use:

```bash
python mebooster/adversarial/dfme_attacker.py
```

Adjust attack parameters in `config.py`, specifying `test_dataset`, `attack_model_arch`, and `victim_model_arch`.

### 4. Constrained Gradient Backpropagation

After executing MAZE and DFME attacks, apply constrained gradient backpropagation by running:

```bash
python mebooster/adversarial/dfme_attacker(constrained_gradient_backpropagation).py
```

This script facilitates experiments with MEBooster and regular fine-tuning.

### 5. Pool-Based Attacks

For pool-based attacks, execute:

```bash
python mebooster/adversarial/pool_based_attacker.py
```

This will perform baseline, width expansion, and MEBooster variations without constrained gradient backpropagation. Results are stored in `models/adversary/ADV_DIR/f{cfg.test_dataset}`, and generated data are saved under `./dfme_data/data_dfme`.
![Uploading image.png…]()

