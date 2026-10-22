# Snackwell update 3: FP32 and tensor execution

This package contains a draft X thread and local media. Nothing has been posted.

Open `review.html` in a browser to review the whole package. `thread-draft.md`
contains seven posts with attachment notes; `single-post.txt` is the opening
post on its own. Copyable image descriptions are in `alt-text.md`.

## Media

| File | Use |
| --- | --- |
| `assets/01-sm-architecture.png` | Opening post: where FP32 and tensor execution fit |
| `assets/02-processing-element.png` | One PE's multiply/add/accumulator data path |
| `assets/03-tmul-registers.png` | TMUL operands, tile layout, and writeback |
| `assets/04-systolic-array.png` | Optional static overview of the 16 PEs |
| `assets/tensor-flow.mp4` | 18.5-second H.264 animation, 1080×1080, 24 fps, no audio |
| `assets/tensor-flow.gif` | Animated GIF alternative, 720×720 |
| `assets/tensor-flow-poster.png` | Poster frame for the animation |
| `assets/05-test-results.png` | Readable typeset view of the fresh regression output |
| `assets/06-testing-screenshot.png` | Cropped browser screenshot of the verbatim captured stdout |

Every diagram also has a matching `.excalidraw` file containing editable text,
boxes, and arrows. Those are local source files; importing project-derived
content into excalidraw.com is awaiting explicit approval after automatic
approval review blocked the initial import. The PNGs were rendered locally.

## Evidence and interpretation

- `source/regression.txt` is unedited stdout from the fresh full regression.
  The command, timestamp, and zero exit status are in `regression-metadata.json`.
- The full-SM tensor suite ran 185 reference matrices, checking 2,960 elements,
  plus directed dependency, mask, warp, divergence, reset, and backpressure cases.
- `source/animation-data.json` contains the animation's A, B, and C matrices.
- `source/verify_animation.py` drove these exact inputs through the implemented
  `systolic_array_4x4`. It checked all 160 PE snapshots across ten logical steps,
  all 64 valid MAC operand pairs, and the final 16-element result. The raw log
  and parsed trace are retained as `animation-rtl.log` and `animation-trace.json`.
- Animation timing is explanatory. The RTL trace advances at clocks 24, 46,
  74, 100, 128, 156, 184, 212, 238, and 250 for this example. Each advance waits
  for all PEs. The video uses equal-duration logical steps to make the movement
  readable and explicitly labels variable clock latency.
- Separate FP32 multiply and add rounding is non-fused. C starts at +0.
- TMUL is fixed 4×4, with one tensor job in flight. FPGA performance, synthesis,
  inference, and training are not claimed.

## Rebuild

Requires Python with Pillow, Icarus Verilog, and macOS with Swift/AVFoundation.
The renderer currently uses macOS system fonts. From this repository:

```sh
swiftc -O docs/social/tensor-update/source/encode_video.swift -o /private/tmp/snackwell-encode-video
python3 docs/social/tensor-update/source/make_media.py
python3 docs/social/tensor-update/source/verify_animation.py
python3 docs/social/tensor-update/source/write_thread.py
python3 docs/social/tensor-update/source/make_review.py
```

`make_media.py --static-only` regenerates the PNG diagrams and Excalidraw sources
without encoding video. Video encoding may need permission to use the system
codec services when run in a restricted sandbox. Intermediate frames live in
`/private/tmp/snackwell-video-frames`; QA snapshots are in `source/qa`.

The screenshot was captured from `testing-log.html` with the browser tool and
cropped only to remove empty margins. It was not redrawn or altered to change
any test results. The separate designed test image is labeled as a typeset view.
