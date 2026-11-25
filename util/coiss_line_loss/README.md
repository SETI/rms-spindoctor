# Cassini ISS lossless-compressor check

Two scripts that test the simulator's model of the Cassini ISS lossless compressor
against real frames. The model is the block-adaptive Rice coder in
`spindoctor.sim.render.telemetry.lossless`: each line is coded as the differences
between neighboring samples, in 16-sample blocks, each block with the cheapest of the
codes that send 0 to 13 low bits as they are, plus 4 bits that name the code. A pair
of lines may use 16 bits for each sample of one line, and the second line of a pair
loses the blocks that do not fit.

Both scripts read the frames listed in `lossless_frames.txt`: the 61 narrow angle
frames and the one wide angle frame, of the losslessly compressed 12-bit frames with
lost lines in COISS_2010, COISS_2023, COISS_2050, and COISS_2068, that lose line ends
to the compressor. A frame qualifies when at least 10 of its lines keep their start
and lose their end and its cut pairs code to at least 0.96 of the budget at the
median. Seven more frames have 10 or more such lines but code to well under the
budget, at a median of 0.3 to 0.7. Their labels give the cause as gaps in the ground
link, and the list leaves them out.

## Layout

| File | Role |
| --- | --- |
| `coiss_raw.py` | Reads a raw image file, its label, and its binary line prefixes, which give each line's valid samples. |
| `lossless_check.py` | Codes every pair whose first line is whole and compares the coded size with the budget. |
| `twins.py` | Builds a simulated twin of a frame and compares the simulator's cuts on the twin with the frame's own. |
| `lossless_frames.txt` | The frames, as paths under the PDS3 holdings root. |

## Running

The holdings root may be a local directory or a URL.

```bash
export PDS3_HOLDINGS_DIR=/path/to/holdings
python util/coiss_line_loss/lossless_check.py
python util/coiss_line_loss/twins.py            # the ten default frames
python util/coiss_line_loss/twins.py N1490874834_1.IMG
```

## Results (2026-11-25)

`lossless_check.py` over the 62 frames:

- 14,038 pairs whose second line lost its end code to 0.970, 0.977, 0.991, 1.002,
  and 1.009 of the budget at the 0.1st, 1st, 50th, 99th, and 99.9th percentiles; 9
  fall outside 0.96 to 1.01.
- 5,245 whole pairs code to at most 1.017 of the budget; 6 exceed it.
- The budget left at a cut is 0.65, 1.17, and 1.34 times the cost of the line's
  last kept blocks at the 50th, 95th, and 99th percentiles.

`twins.py` builds a twin from the frame's lines with their lost ends filled from
the line above, smoothed by a 9 x 9 median filter, plus Gaussian noise whose local
spread matches the frame's differences between neighboring samples about that
smooth brightness. Over the ten default frames the twins cut the same sections as
their frames, their median lost share is within one block of the real one on nine
and within two on the tenth, and their spread of lost share within a frame is
within 35% of the real one. Where the content is sparse a twin cuts more of its
pairs than its frame: 29% against 20%, 48% against 31%, and 13% against 4%.
