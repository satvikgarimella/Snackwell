"""Execute and verify transformer prefill, cached decode and evaluation on RTL."""

import json
import math
import struct
import time
from pathlib import Path

from snackwell.character_model import block_prefix, feedforward_width, reference_forward
from snackwell.decoding import validate_temperature
from snackwell.kv_cache import model_signature
from snackwell.model_runtime import ModelRuntime, insert_rectangle, rectangle
from snackwell.numeric import bits, value
from snackwell.reference.inference import stable_softmax, verify_approx
from snackwell.simulator import ModelSimulator, SimulatorSession, compile_model_runner


def forward(
    model,
    ids,
    output,
    executable=None,
    on_progress=None,
    draw=None,
    last_only=False,
    temperature=1.0,
    kv_cache=None,
    phase=None,
):
    """Verify a forward pass, releasing its simulator process even on cancellation."""
    (Path(output) / "result.json").unlink(missing_ok=True)
    if kv_cache is not None:
        if executable is None:
            executable = compile_model_runner(Path(output))
        if not getattr(executable, "source_sha256", None):
            raise ValueError("KV caching requires a model runner with an RTL source fingerprint")
    if isinstance(executable, ModelSimulator) and executable.backend == "verilator":
        with SimulatorSession(executable) as session:
            return _forward(
                model,
                ids,
                output,
                session,
                on_progress,
                draw,
                last_only,
                temperature,
                kv_cache,
                phase,
            )
    return _forward(
        model, ids, output, executable, on_progress, draw, last_only, temperature, kv_cache, phase
    )


def _forward(
    model, ids, output, executable, on_progress, draw, last_only, temperature, kv_cache, phase
):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    validate_temperature(temperature)
    if draw is not None and (not math.isfinite(draw) or not 0 <= value(bits(draw)) < 1):
        raise ValueError("sampling draw must round to FP32 in [0, 1)")
    config = model["config"]
    context, width, vocabulary_size = len(ids), config["width"], len(config["vocabulary"])
    start = len(kv_cache.ids) if kv_cache is not None else 0
    signature = model_signature(model) if kv_cache is not None else None
    rtl_digest = getattr(executable, "source_sha256", None)
    if start and (
        tuple(ids[:-1]) != kv_cache.ids
        or context != start + 1
        or kv_cache.signature != signature
        or kv_cache.rtl_source_sha256 != rtl_digest
        or len(kv_cache.layers) != config.get("layers", 1)
        or any(len(k) != start * width or len(v) != start * width for k, v in kv_cache.layers)
        or kv_cache.reference is None
        or kv_cache.reference.ids != kv_cache.ids
    ):
        raise ValueError("KV cache does not match this prefix, checkpoint or RTL build")
    # This state contains only independent double-precision reference results.
    # Commit it together with RTL state only after every check succeeds.
    reference_started = time.perf_counter()
    expected, reference_state = reference_forward(
        model, ids, previous=kv_cache.reference if start else None, return_state=True
    )
    reference_seconds = time.perf_counter() - reference_started
    new_tokens = context - start
    input_rows = 1 if new_tokens == 1 else max(4, (new_tokens + 3) // 4 * 4)
    attention_columns = max(4, 1 << (context - 1).bit_length())
    pending_cache = []
    hidden_width = feedforward_width(config)
    padded_vocabulary = max(4, 1 << (vocabulary_size - 1).bit_length())
    # Encode each validated tensor in one call, retaining the same FP32 rounding.
    weights = {
        name: list(struct.unpack(f"<{len(data)}I", struct.pack(f"<{len(data)}f", *data)))
        for name, data in model["weights"].items()
    }
    runtime = ModelRuntime(output, executable, on_progress)
    checks = {}

    def check(name, actual, row_offset=0, columns=None):
        columns = columns or width
        actual = actual[: (context - row_offset) * columns]
        reference = expected[name][row_offset * columns :]
        # Threshold fixed before checkpoint training; failures are not relaxed.
        try:
            checks[name] = verify_approx(actual, reference, "transformer")
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
        (output / f"{name}.json").write_text(json.dumps([value(a) for a in actual]) + "\n")
        # Verification ignores transport rows; callers retain the padded tensor.

    def norm(name, x, suffix, rows, prefix=""):
        return runtime.rows(
            name,
            "layernorm",
            x,
            rows,
            width,
            gamma=weights[prefix + "gamma" + suffix],
            beta=weights[prefix + "beta" + suffix],
            epsilon=config["epsilon"],
        )

    def linear(name, x, weight, bias, inner, columns, rows):
        return runtime.matmul(
            name + "_matmul", x, weights[weight], rows, columns, inner, bias=weights[bias]
        )

    x = runtime.lookup(
        ids[start:] + [0] * (input_rows - new_tokens),
        weights["token_embedding"],
        weights["position_embedding"][start * width : context * width]
        + [0] * ((input_rows - new_tokens) * width),
        width,
        vocabulary_size,
    )
    check("embedding", x, start, width)
    layers, heads = config.get("layers", 1), config.get("heads", 1)
    head_width = width // heads
    for layer in range(layers):
        prefix = block_prefix(layer, layers)
        # Prefill needs earlier hidden states to form deeper-layer K/V. Decode
        # already has them cached, so input_rows is just one new row.
        final_row = last_only and layer == layers - 1
        rows = 1 if final_row else input_rows
        local_offset = new_tokens - 1 if final_row else 0
        row_offset = start + local_offset
        normalized = norm(prefix + "layernorm1", x, "1", input_rows, prefix)
        if input_rows == 1:
            # Concatenate output columns only. Each dot product and bias uses
            # exactly the same RTL arithmetic, but fewer column-chunk launches.
            qkv_weights = [
                word
                for r in range(width)
                for name in ("q", "k", "v")
                for word in weights[prefix + "w" + name][r * width : (r + 1) * width]
            ]
            qkv_bias = [word for name in ("q", "k", "v") for word in weights[prefix + "b" + name]]
            qkv = runtime.matmul(
                prefix + "qkv", normalized, qkv_weights, 1, 3 * width, width, bias=qkv_bias
            )
            q, k, values = [qkv[i * width : (i + 1) * width] for i in range(3)]
        else:
            q = linear(
                prefix + "q",
                normalized[local_offset * width : (local_offset + rows) * width],
                prefix + "wq",
                prefix + "bq",
                width,
                width,
                rows,
            )
            k, values = [
                linear(
                    prefix + name,
                    normalized,
                    prefix + "w" + name,
                    prefix + "b" + name,
                    width,
                    width,
                    input_rows,
                )
                for name in ("k", "v")
            ]
        k, values = k[: new_tokens * width], values[: new_tokens * width]
        if start:
            old_k, old_v = kv_cache.layers[layer]
            k, values = list(old_k) + k, list(old_v) + values
        pending_cache.append((tuple(k), tuple(values)))
        k += [0] * ((attention_columns - context) * width)
        values += [0] * ((attention_columns - context) * width)
        attended = [None] * (rows * width)
        for head in range(heads):
            column = head * head_width
            query = rectangle(q, width, 0, rows, column, head_width)
            key = rectangle(k, width, 0, attention_columns, column, head_width)
            val = rectangle(values, width, 0, attention_columns, column, head_width)
            name = prefix + (f"head{head}_" if heads > 1 else "")
            scores = runtime.matmul(
                name + "attention_qk",
                query,
                key,
                rows,
                attention_columns,
                head_width,
                transpose_b=True,
            )
            scores = runtime.attention_scale(
                scores, rows, head_width, attention_columns, row_offset
            )
            probabilities = runtime.rows(
                name + "attention_softmax", "softmax", scores, rows, attention_columns
            )
            if on_progress and layer == layers - 1 and head == 0:
                last_query = (context - row_offset - 1) * attention_columns
                on_progress(
                    dict(
                        type="attention",
                        head=head,
                        weights=[
                            value(word) for word in probabilities[last_query : last_query + context]
                        ],
                    )
                )
            product = runtime.matmul(
                name + "attention_pv", probabilities, val, rows, head_width, attention_columns
            )
            insert_rectangle(attended, width, 0, rows, column, head_width, product)
        projection = linear(
            prefix + "attention_output", attended, prefix + "wo", prefix + "bo", width, width, rows
        )
        residual = runtime.rows(
            prefix + "attention_residual",
            "residual_add",
            x[local_offset * width : (local_offset + rows) * width],
            rows,
            width,
            other=projection,
        )
        hidden = linear(
            prefix + "ffn_up",
            norm(prefix + "layernorm2", residual, "2", rows, prefix),
            prefix + "w1",
            prefix + "b1",
            width,
            hidden_width,
            rows,
        )
        hidden = runtime.rows(prefix + "ffn_gelu", "gelu", hidden, rows, hidden_width)
        projection = linear(
            prefix + "ffn_down", hidden, prefix + "w2", prefix + "b2", hidden_width, width, rows
        )
        x = runtime.rows(
            prefix + "ffn_residual", "residual_add", residual, rows, width, other=projection
        )
        check(prefix + "block", x, row_offset, width)
    normalized = norm("final_norm", x, "_final", rows)
    check("final_norm", normalized, row_offset, width)
    output_rows = context - row_offset
    # TMUL needs four output columns; softmax needs a power of two. Compute
    # only the former, then copy into the latter layout before RTL masking.
    projection_columns = (vocabulary_size + 3) // 4 * 4
    weight = [
        weights["lm_weight"][r * vocabulary_size + col] if col < vocabulary_size else 0
        for r in range(width)
        for col in range(projection_columns)
    ]
    logits = runtime.matmul(
        "lm_head",
        normalized,
        weight,
        rows,
        projection_columns,
        width,
        bias=weights["lm_bias"] + [0] * (projection_columns - vocabulary_size),
    )
    if projection_columns != padded_vocabulary:
        padded = [0] * (rows * padded_vocabulary)
        insert_rectangle(padded, padded_vocabulary, 0, rows, 0, projection_columns, logits)
        logits = padded
    check(
        "logits",
        rectangle(logits, padded_vocabulary, 0, rows, 0, vocabulary_size),
        row_offset,
        vocabulary_size,
    )
    logits = runtime.rows(
        "vocabulary_mask",
        "vocabulary_mask",
        logits,
        rows,
        padded_vocabulary,
        vocabulary=vocabulary_size,
    )
    probability_bits = runtime.rows("output_softmax", "softmax", logits, rows, padded_vocabulary)
    probabilities = rectangle(
        probability_bits, padded_vocabulary, 0, output_rows, 0, vocabulary_size
    )
    check("probabilities", probabilities, row_offset, vocabulary_size)
    if any(
        probability_bits[r * padded_vocabulary + col] != 0
        for r in range(rows)
        for col in range(vocabulary_size, padded_vocabulary)
    ):
        raise ValueError("padded vocabulary has nonzero probability")
    predictions = runtime.rows(
        "greedy_decode", "argmax", probability_bits, rows, padded_vocabulary
    )[:output_rows]
    if predictions != expected["predictions"][row_offset:]:
        raise ValueError("RTL greedy predictions differ from whole-model reference")
    selected = predictions[-1]
    sampling_probabilities = None
    if draw is not None:
        # Sample only the final real row, including for partial prompt tiles.
        last_row = slice((output_rows - 1) * padded_vocabulary, output_rows * padded_vocabulary)
        sampling_bits = probability_bits[last_row]
        if temperature != 1.0:
            sampling_bits = runtime.rows(
                "sampling_softmax",
                "softmax",
                logits[last_row],
                1,
                padded_vocabulary,
                temperature=temperature,
            )
            sampling_expected = [
                p
                for row in [context - 1]
                for p in stable_softmax(
                    [
                        z / temperature
                        for z in expected["logits"][
                            row * vocabulary_size : (row + 1) * vocabulary_size
                        ]
                    ]
                )
            ]
            checks["sampling_probabilities"] = verify_approx(
                rectangle(sampling_bits, padded_vocabulary, 0, 1, 0, vocabulary_size),
                sampling_expected,
                "transformer",
            )
        if any(
            sampling_bits[r * padded_vocabulary + col] != 0
            for r in range(1)
            for col in range(vocabulary_size, padded_vocabulary)
        ):
            raise ValueError("sampling assigned probability to padded vocabulary")
        sampling_probabilities = [
            value(word) for word in sampling_bits[-padded_vocabulary:][:vocabulary_size]
        ]
        selected = runtime.rows(
            "sample_decode",
            "categorical",
            sampling_bits,
            1,
            padded_vocabulary,
            draws=[bits(draw)],
        )[-1]
    selected_distribution = (
        sampling_probabilities
        if draw is not None
        else [value(word) for word in probabilities[-vocabulary_size:]]
    )
    if not 0 <= selected < vocabulary_size or selected_distribution[selected] <= 0:
        raise ValueError("decoder selected a masked or zero-probability character")
    report = dict(
        verified=True,
        simulation=runtime.simulation,
        rtl_source_sha256=getattr(executable, "source_sha256", None),
        wall_seconds=time.perf_counter() - started,
        reference_seconds=reference_seconds,
        checkpoint_sha256=model.get("sha256"),
        input_ids=ids,
        output_row_offset=row_offset,
        output_rows=output_rows,
        last_only=last_only,
        cache=dict(
            enabled=kv_cache is not None,
            phase=phase or ("decode" if start else "prefill"),
            reused_tokens=start,
            computed_tokens=new_tokens,
            context_tokens=context,
            storage="host raw FP32 words" if kv_cache is not None else None,
            bytes=2 * layers * context * width * 4 if kv_cache is not None else 0,
        ),
        cycles=sum(r["cycles"] for r in runtime.reports),
        launches=len(runtime.reports),
        predictions=predictions,
        next_character=config["vocabulary"][selected],
        greedy_next_character=config["vocabulary"][predictions[-1]],
        selection=dict(
            mode="sample" if draw is not None else "greedy",
            draw=draw,
            token_id=selected,
            temperature=temperature if draw is not None else None,
        ),
        sampling_probabilities=sampling_probabilities,
        probabilities=[value(word) for word in probabilities],
        accuracy=checks,
        stages=runtime.reports,
    )
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    if kv_cache is not None:
        kv_cache.ids = tuple(ids)
        kv_cache.layers = tuple(pending_cache)
        kv_cache.signature = signature
        kv_cache.rtl_source_sha256 = rtl_digest
        kv_cache.reference = reference_state
    return report


def evaluate(model, examples_path, output, executable):
    examples = json.loads(Path(examples_path).read_text())
    if examples.get("checkpoint_sha256") != model["sha256"]:
        raise ValueError("evaluation examples belong to a different checkpoint")
    cases = examples.get("examples", [])
    if not cases:
        raise ValueError("evaluation set is empty")
    loss, baseline, correct, count, cycles = 0.0, 0.0, 0, 0, 0
    reports = []
    for index, case in enumerate(cases):
        ids, targets = case["input_ids"], case["target_ids"]
        v = len(model["config"]["vocabulary"])
        if len(targets) != len(ids) or any(type(t) is not int or not 0 <= t < v for t in targets):
            raise ValueError("invalid evaluation targets")
        report = forward(model, ids, output / f"example_{index:03d}", executable)
        metrics = verify_approx(
            [bits(p) for p in report["probabilities"]], case["torch_probabilities"], "transformer"
        )
        for row, target in enumerate(targets):
            probability = report["probabilities"][row * v + target]
            if probability <= 0:
                raise ValueError("target probability underflowed to zero")
            loss -= math.log(probability)
            baseline -= math.log(examples["unigram_probabilities"][target])
            correct += report["predictions"][row] == target
            count += 1
        cycles += report["cycles"]
        reports.append(
            dict(index=index, verified=True, pytorch_accuracy=metrics, cycles=report["cycles"])
        )
        print(
            f"PASS held-out example {index + 1}/{len(cases)}: {report['cycles']} cycles", flush=True
        )
    if loss >= baseline:
        raise ValueError("held-out RTL loss did not improve on the unigram baseline")
    return dict(
        verified=True,
        checkpoint_sha256=model["sha256"],
        evaluated_tokens=count,
        loss=loss / count,
        perplexity=math.exp(loss / count),
        accuracy=correct / count,
        unigram_loss=baseline / count,
        improves_unigram=True,
        cycles=cycles,
        examples=reports,
    )
