# Inference assembly examples

These are default shape-specialized assembly programs generated from
`scripts/inference_kernels.py`. Their headers list the required array and
constant addresses. The runner supplies those data images and verifies results:

```sh
python3 scripts/run_kernel.py all
python3 scripts/run_kernel.py softmax --rows 8 --cols 16
python3 scripts/run_kernel.py attention --rows 8 --cols 8
```

Edit the emitters for kernel changes, then run `python3 scripts/write_kernel_examples.py`.
Every simulation also saves its exact generated source as `<stage>/program.asm`.
See [the kernel guide](../../docs/INFERENCE_KERNELS.md) for algorithms, input
formats, numerical tolerances, limitations, and tests.
