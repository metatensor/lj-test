# Lennard-Jones potential as a metatensor atomistic model

This repository contains an implementation of the classic Lennard-Jones
potential as a [metatensor atomistic
model](https://lab-cosmo.github.io/metatensor/latest/atomistic/index.html).

This is intended to be used when testing the integration of metatensor atomistic
models with various simulation engines; but can also serve as a relatively
complete example of how to write your own models and use neighbor lists.

### Installation

```bash
pip install git+https://github.com/Luthaf/metatensor-lj-test

# if you are working on Linux or Windows and DO NOT have a CUDA GPU
pip install --extra-index-url=https://download.pytorch.org/whl/cpu git+https://github.com/Luthaf/metatensor-lj-test
```

### API

```python
import metatensor_lj_test

# `with_extension` controls wether the model uses custom
# TorchScript operators or a pure PyTorch implementation
model = metatensor_lj_test.lennard_jones_model(
    atomic_type=12,
    cutoff=3.4,
    sigma=1.5,
    epsilon=23.0,
    length_unit="Angstrom",
    energy_unit="eV",
    with_extension=False,
)

model.export("lennard-jones.pt", collect_extension="extensions/")
```
