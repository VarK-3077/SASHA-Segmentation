"""
List of CAMELYON16 training tumor slides with known non-exhaustive pixel-level
annotations: their normal-looking regions may still hide unannotated tumor, so they
must be excluded when sampling negative (normal) patches from tumor slides.

The NCRF repo (github.com/baidu-research/NCRF) README does not carry this list
(checked 2026-08-25 against its raw README.md: no mention of exhaustive/partial/
excluded annotations besides the unrelated Test_049/Test_114 FROC-evaluation
exclusion). Wang et al. 2016, the Camelyon16 winning-entry paper (arXiv:1606.05718),
doesn't mention it either. The actual documented source is Liu et al. 2017, "Detecting
Cancer Metastases on Gigapixel Pathology Images" (Google), Appendix A.4.

Quoted verbatim from https://arxiv.org/abs/1703.02442 (Appendix A.4, "Tumor slides
with incomplete annotations"):

    At the outset, 11 tumor slides were known to have non-exhaustive pixel level
    annotations: 015, 018, 020, 029, 033, 044, 046, 051, 054, 055, 079, 092, and 095.
    Thus, we did not use non-tumor patches from these slides as training examples of
    normal patches. Over the course of our experiments, we discovered several more
    such cases that we verified with a pathologist: 010, 025, 034, 056, 067, 085, 110.

Note: the paper says "11 tumor slides" but enumerates 13 numbers in that first
group (015, 018, 020, 029, 033, 044, 046, 051, 054, 055, 079, 092, 095) - that
mismatch is in the source paper itself, not a transcription error here. The list
below is the union of both groups: 13 + 7 = 20 slides.
"""

SOURCE_URL = 'https://arxiv.org/abs/1703.02442'

QUOTE = (
    'At the outset, 11 tumor slides were known to have non-exhaustive pixel level '
    'annotations: 015, 018, 020, 029, 033, 044, 046, 051, 054, 055, 079, 092, and 095. '
    'Thus, we did not use non-tumor patches from these slides as training examples of '
    'normal patches. Over the course of our experiments, we discovered several more '
    'such cases that we verified with a pathologist: 010, 025, 034, 056, 067, 085, 110.'
)

_INITIAL = [15, 18, 20, 29, 33, 44, 46, 51, 54, 55, 79, 92, 95]
_DISCOVERED = [10, 25, 34, 56, 67, 85, 110]

NON_EXHAUSTIVE_TUMOR_SLIDES = sorted(f'tumor_{n:03d}' for n in _INITIAL + _DISCOVERED)
