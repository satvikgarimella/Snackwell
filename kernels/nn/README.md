# Inference assembly examples

These are default shape-specialized assembly programs generated from
`snackwell/kernels/inference.py` and `snackwell/kernels/transformer.py`. Their headers list the required array and
constant addresses. The runner supplies those data images and verifies results:

```sh
python3 scripts/run_kernel.py all
python3 scripts/run_kernel.py softmax --rows 8 --cols 16
python3 scripts/run_kernel.py attention --rows 8 --cols 8
python3 scripts/run_kernel.py transformer
```

Edit the emitters for kernel changes, then run `python3 scripts/write_kernel_examples.py`.
Every simulation also saves its exact generated source as `<stage>/program.asm`.
See [the kernel guide](../../docs/INFERENCE_KERNELS.md) for algorithms, input
formats, numerical tolerances, limitations, and tests.
The 21 files under `transformer/` form the sequential launches described in
[the transformer guide](../../docs/TRANSFORMER.md); each consumes the actual
memory left by the preceding stage.
