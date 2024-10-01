# H-embedding guided Hypernet

## Setup Python Environment

- Use conda to build environment:

```console
$ conda env create -f environment.yml
$ conda activate hypercl_env
```
- Set paths to your working directory and data directory correctly

## Run experiment for Cifar10/100

- Our method:  
  Simply run the `run_resnet.sh` file under the cifar directory. The arguments are now set to conform with the results in paper, you may modify them as you wish.

- Baselines:
  Currently we only support the ablation baselines, which could be derived respectively by changing the `--emb_metric` argument from `Hembedding` to `random`, and disabling `--emb_reg`
