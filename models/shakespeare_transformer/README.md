# Broader-corpus Shakespeare model

Eight transformer blocks, four attention heads, width 128, feed-forward width
512, context 128 subword tokens, vocabulary 512; 1,734,400 parameters. The weights
were trained with PyTorch on Apple MPS. Snackwell executes them separately in RTL
simulation; the hardware verification record is `verification.json`.

The full 128-token forward pass, context rebuild, cached decode and a complete
RTL training step passed. All 1,734,400 parameter gradients and FP32 updates
were checked, followed by checkpoint reload and inference from the updated
weights. Those audit updates are kept separately from this inference checkpoint.

This model lowers matched validation token loss by 9.06% compared with
`models/subword_transformer/checkpoint.json`. It still makes grammatical and
semantic errors, and greedy sentence loops worsened in the fixed evaluation.
The sampled cases do not establish a reduction in repetition. It continues
Shakespeare-style text and was not trained to answer questions or follow
instructions. The existing default is preserved.

From the repository root:

```sh
python3 scripts/run_model.py \
  --checkpoint models/shakespeare_transformer/checkpoint.json \
  --interactive --tokens 32 --seed 92617660 --temperature 0.7
```

Enter `My lord, I have something to say` or `ROMEO:`. Tokens can contain multiple
characters. The terminal should show checkpoint `28e378b741fa`. This command uses
the actual RTL runner, with numerical verification enabled. The larger model
requires more computation than the existing default. In the recorded 32-token
run, prompt processing took 22.26 seconds and cached decode averaged 6.28 seconds
per token, versus 7.59 seconds and 2.55 seconds for the previous subword model.

For a fast CPU-only language preview, using the optional PyTorch environment:

```sh
build/model-venv/bin/python scripts/preview_model.py \
  --checkpoint models/shakespeare_transformer/checkpoint.json \
  --interactive --tokens 64 --seed 92617660 --temperature 0.7
```

The preview does not execute Snackwell. It is useful for inspecting long
continuations before running the hardware simulation.

`evidence/` retains the fixed complete sample sets, matched evaluation targets,
training recipe, corpus manifest and regression log. The supplement excludes
the entire held-out play and leaves the original tokenizer and evaluation splits
unchanged. See [the experiment report](../../docs/QUALITY_REPAIR.md) for the
comparison, data preparation, failed alternatives and limitations.
