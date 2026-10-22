# GELU network trained on Snackwell

This 8→16→4 network was initialized randomly and trained for 200 SGD updates by
the simulated SM. It learns four nonlinear numeric functions; it is separate
from the character transformer.

- `initial_checkpoint.json`: the seed-1729 starting parameters.
- `checkpoint.json`: the 212 parameter words after 200 verified RTL updates.
- `verification.json`: measured losses, cycle counts and source/checkpoint identities.

Run the complete training experiment again:

```sh
python3 scripts/train_snackwell.py
```

Continue these saved weights for another 100 steps:

```sh
python3 scripts/train_snackwell.py --resume models/gelu_network/checkpoint.json --steps 100
```

Training loss fell from 0.172689 to 0.033176 and held-out loss from 0.137247 to
0.038508. The new run directory contains all four predicted outputs, intermediate
activations, gradients, updated parameters and numerical checks. All tensor
values in checkpoints are raw unsigned 32-bit IEEE 754 words, preserving exact
RTL outputs for resume. These files use the `snackwell.ffn-sgd.v1` schema and are
not accepted by the character-model inference runner.

See [training notes](../../docs/TRAINING.md) for the data, equations and remaining
transformer-training work.
