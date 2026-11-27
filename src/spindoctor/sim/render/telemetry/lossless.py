"""A block-adaptive Rice coder of line pairs, and the cut a fixed pair budget forces.

A lossless line compressor of the Rice family codes each line as the differences
between horizontally adjacent samples.  It splits the differences into blocks of a
fixed number of samples and codes each block with whichever of its code options
spends the fewest bits on it.  Option ``k`` sends the low ``k`` bits of each mapped
difference as they are and the rest of it in unary, so a block of ``n`` samples whose
mapped differences are ``m`` costs ``n * (k + 1) + sum(m >> k)`` bits, plus the bits
that name the option.  A difference ``d`` maps to ``2 * d`` when it is not negative
and to ``-2 * d - 1`` when it is, so small differences of either sign cost little.
The first sample of a line is the reference the differences start from, and it costs
nothing here.

A camera whose compressor must fit each pair of lines into a fixed number of bits
codes the first line of the pair, then the second line's blocks in order while they
still fit; the decoder keeps whole blocks only, so the second line keeps a whole
number of blocks from its start and loses the rest.  Where the content is smooth the
pair fits and nothing is lost; where it carries noise or fine detail the second line
loses its end, and neighboring pairs, which carry nearly the same content, lose
nearly the same share.

:func:`block_bits` codes the blocks of every line of a frame, and :func:`pair_cuts`
finds what each pair keeps within its budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spindoctor.support.types import NDArrayIntType

__all__ = ['PairCuts', 'block_bits', 'pair_cuts']


def block_bits(
    dn: NDArrayIntType, *, block_samples: int, id_bits: int, max_split: int
) -> NDArrayIntType:
    """Return the coded size in bits of every block of every line.

    Parameters:
        dn: The integer samples, one line per row.
        block_samples: The samples in a block.  A last block shorter than this, where
            the line length is not a whole number of blocks, is coded at its own
            length.
        id_bits: The bits that name a block's code option.
        max_split: The largest number of low bits an option sends as they are; the
            options run from 0 to this.

    Returns:
        The bits of each block, one row per line and one column per block.
    """
    x = np.asarray(dn, dtype=np.int64)
    rows, samples = x.shape
    diff = np.diff(x, axis=1, prepend=x[:, :1])
    mapped = np.where(diff >= 0, 2 * diff, -2 * diff - 1)
    blocks = -(-samples // block_samples)
    padded = np.zeros((rows, blocks * block_samples), dtype=np.int64)
    padded[:, :samples] = mapped
    grouped = padded.reshape(rows, blocks, block_samples)
    lengths = np.full(blocks, block_samples, dtype=np.int64)
    lengths[-1] = samples - (blocks - 1) * block_samples
    best = np.full((rows, blocks), np.iinfo(np.int64).max, dtype=np.int64)
    for split in range(max_split + 1):
        cost = lengths * (split + 1) + (grouped >> split).sum(axis=2)
        np.minimum(best, cost, out=best)
    result: NDArrayIntType = best + id_bits
    return result


@dataclass(frozen=True)
class PairCuts:
    """What each line pair keeps within its budget.

    Parameters:
        first_kept: The blocks the first line of each pair keeps; all of them unless the
            first line alone exceeds the budget.
        second_kept: The blocks the second line of each pair keeps.
        pair_bits: The coded size of the blocks each pair keeps.
    """

    first_kept: NDArrayIntType
    second_kept: NDArrayIntType
    pair_bits: NDArrayIntType


def pair_cuts(bits: NDArrayIntType, *, budget_bits: int) -> PairCuts:
    """Find the blocks each pair of lines keeps within its budget.

    The pairs are lines 0 and 1, 2 and 3, and so on; a last line without a partner is
    left out.  The first line of a pair is coded whole, then the second line's blocks
    in order while the running total stays within ``budget_bits``.  If the first line
    alone exceeds the budget, it keeps the blocks that fit and the second line keeps
    none.

    Parameters:
        bits: The bits of each block, from :func:`block_bits`.
        budget_bits: The bits a pair may use.

    Returns:
        The blocks each pair keeps and their coded size.
    """
    pairs = bits.shape[0] // 2
    first = np.asarray(bits[0 : 2 * pairs : 2], dtype=np.int64)
    second = np.asarray(bits[1 : 2 * pairs : 2], dtype=np.int64)
    first_cum = np.cumsum(first, axis=1)
    first_total = first_cum[:, -1]
    second_cum = first_total[:, None] + np.cumsum(second, axis=1)
    first_kept = np.count_nonzero(first_cum <= budget_bits, axis=1)
    # A first line over the budget leaves the second none: its running total starts there.
    second_kept = np.count_nonzero(second_cum <= budget_bits, axis=1)
    rows = np.arange(pairs)
    first_bits = np.where(first_kept > 0, first_cum[rows, np.maximum(first_kept - 1, 0)], 0)
    second_bits = np.where(
        second_kept > 0,
        second_cum[rows, np.maximum(second_kept - 1, 0)] - first_total,
        0,
    )
    return PairCuts(
        first_kept=first_kept, second_kept=second_kept, pair_bits=first_bits + second_bits
    )
