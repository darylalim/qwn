"""Answer post-processing. Phase 0 needs only citation parsing; phase 2 adds the answer flow."""

import logging
import re

logger = logging.getLogger(__name__)

_CITATION = re.compile(r"\[(S\d+)\]")


def parse_citations(text: str, labels: list[str]) -> list[str]:
    """Labels cited in `text`, in order of first appearance. Invented labels are dropped."""
    cited: list[str] = []
    invented: set[str] = set()
    for label in _CITATION.findall(text):
        if label not in labels:
            invented.add(label)
        elif label not in cited:
            cited.append(label)
    if invented:
        logger.warning("dropped invented citations: %s", sorted(invented))
    return cited
