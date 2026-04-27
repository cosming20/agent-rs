"""Drop-in formula decoder that uses pix2tex (80M params) instead of
Docling's CodeFormulaV2 (1.8B params).

Why this module exists:

Docling's built-in ``do_formula_enrichment`` runs a general-purpose
Idefics3 VLM that takes ~60 s per formula on CPU / Apple Silicon.
pix2tex is a specialist ViT encoder + transformer decoder trained
exclusively on rendered LaTeX images. On clean printed math it matches
CodeFormulaV2's accuracy at ~1-2 s per formula on CPU — roughly a
30-50× wall-clock speedup for the hot path of the pipeline.

Integration point: we hook **after** Docling's ``DocumentConverter``
has produced a ``DoclingDocument`` but **before** the ``HybridChunker``
runs. We walk the doc's formula items, crop each formula's bounding
box from its page image, run pix2tex, and write the decoded LaTeX
back into the item's ``.text`` attribute. Chunking downstream sees
real LaTeX instead of the ``<!-- formula-not-decoded -->`` sentinel.

Activation via ``DOCLING_FORMULA_BACKEND`` env var:
  - ``pix2tex``      — this module (default)
  - ``codeformula``  — Docling's native 1.8B VLM (slow, needs GPU)
  - ``off``          — skip entirely; formulas stay as placeholder
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FormulaStats:
    """Diagnostic counters emitted once per document."""

    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    total_latency_ms: int = 0

    @property
    def avg_ms(self) -> float:
        return self.total_latency_ms / self.succeeded if self.succeeded else 0.0


class Pix2TexEnricher:
    """Lazy-loaded pix2tex wrapper with a Docling-document enrichment method.

    The underlying ``LatexOCR`` model is heavy to construct (~200 MB of
    weights), so we defer its creation until the first formula we
    actually need to decode. Documents with zero formulas pay zero cost.
    """

    def __init__(self) -> None:
        self._model: Any | None = None

    # ------------------------------------------------------------------
    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        logger.info("Loading pix2tex model (first use — downloads ~200 MB on cold cache)")
        t0 = time.monotonic()
        # Deferred import so the sidecar starts quickly when the user
        # has selected a different backend.
        from pix2tex.cli import LatexOCR  # type: ignore[import]

        self._model = LatexOCR()
        logger.info("pix2tex loaded in %.2fs", time.monotonic() - t0)
        return self._model

    # ------------------------------------------------------------------
    def enrich_document(self, doc: Any) -> FormulaStats:
        """Fill LaTeX text into every formula item of ``doc`` in place.

        ``doc`` is a ``docling_core.types.doc.DoclingDocument``. We rely
        only on the public surface (``.pages``, ``.texts``, ``.iterate_items``)
        to stay robust across minor Docling version bumps.
        """
        stats = FormulaStats()
        formula_items = list(self._find_formula_items(doc))
        if not formula_items:
            return stats

        # Resolve page images once (Docling's PdfPipelineOptions must have
        # ``generate_page_images=True``). Build a page_no → PIL.Image map
        # so we don't re-walk doc.pages for each formula.
        page_images: dict[int, Any] = {}
        for page_no, page in getattr(doc, "pages", {}).items():
            img_ref = getattr(page, "image", None)
            if img_ref is None:
                continue
            pil = getattr(img_ref, "pil_image", None)
            if pil is not None:
                page_images[int(page_no)] = pil

        if not page_images:
            logger.warning(
                "pix2tex enrichment: %d formula items found but no page images "
                "were rendered — check that `generate_page_images=True` is set in "
                "PdfPipelineOptions",
                len(formula_items),
            )
            return stats

        model = self._ensure_model()

        for item in formula_items:
            stats.attempted += 1
            bbox, page_no = _first_prov_bbox(item)
            if bbox is None or page_no is None:
                logger.debug("formula item missing bbox/page_no — skipping")
                stats.failed += 1
                continue

            page_img = page_images.get(page_no)
            if page_img is None:
                logger.debug("no page image for page %d — skipping formula", page_no)
                stats.failed += 1
                continue

            try:
                crop = _crop_bbox(page_img, page_no, doc, bbox)
            except Exception as exc:  # noqa: BLE001
                logger.debug("failed to crop formula on page %d: %s", page_no, exc)
                stats.failed += 1
                continue

            t0 = time.monotonic()
            try:
                latex = model(crop)
            except Exception as exc:  # noqa: BLE001
                # pix2tex occasionally raises on degenerate inputs
                # (empty crops, pure-white areas). Leave the placeholder
                # text alone so the chunk still surfaces the formula's
                # position — better than losing the signal entirely.
                logger.warning("pix2tex failed on page %d: %s", page_no, exc)
                stats.failed += 1
                continue

            latency_ms = int((time.monotonic() - t0) * 1000)
            stats.total_latency_ms += latency_ms

            # Write the decoded LaTeX back. Wrap in `$…$` so the
            # downstream markdown/chunker sees it as math the same way
            # Docling would with CodeFormulaV2.
            if latex:
                item.text = f"$${latex}$$"
                stats.succeeded += 1
            else:
                stats.failed += 1

        logger.info(
            "pix2tex enrichment: %d/%d formulas decoded (avg %.1f ms, %d failed)",
            stats.succeeded,
            stats.attempted,
            stats.avg_ms,
            stats.failed,
        )
        return stats

    # ------------------------------------------------------------------
    @staticmethod
    def _find_formula_items(doc: Any):
        """Yield every formula item in the document.

        Docling's FormulaItem subclasses ``TextItem`` and carries label
        ``DocItemLabel.FORMULA``. We match by label string rather than
        ``isinstance`` to avoid importing Docling's private module paths
        (which move across versions).
        """
        if not hasattr(doc, "iterate_items"):
            return
        for item, _level in doc.iterate_items():
            label = getattr(item, "label", None)
            label_str = str(getattr(label, "value", label) or "").lower()
            if "formula" in label_str:
                yield item


# ----------------------------------------------------------------------
# bbox / crop helpers
# ----------------------------------------------------------------------


def _first_prov_bbox(item: Any) -> tuple[Any | None, int | None]:
    """Return ``(bbox, page_no)`` from the first provenance entry, if any."""
    provs = getattr(item, "prov", None) or []
    if not provs:
        return None, None
    prov = provs[0]
    bbox = getattr(prov, "bbox", None)
    page_no = getattr(prov, "page_no", None)
    return bbox, int(page_no) if page_no is not None else None


def _crop_bbox(page_img: Any, page_no: int, doc: Any, bbox: Any):
    """Crop a Docling bbox out of a page image.

    Docling bboxes live in a coordinate system that may be in PDF
    points (BOTTOMLEFT origin) rather than image pixels (TOPLEFT
    origin). We normalise to pixel-top-left here so PIL's ``.crop``
    does the right thing.
    """
    from PIL import Image  # noqa: F401  (check install early)

    l = float(getattr(bbox, "l", 0.0))
    t = float(getattr(bbox, "t", 0.0))
    r = float(getattr(bbox, "r", 0.0))
    b = float(getattr(bbox, "b", 0.0))

    page = None
    pages = getattr(doc, "pages", None)
    if pages is not None:
        page = pages.get(page_no)

    # Convert to image pixel coordinates. Docling exposes the bbox's
    # coordinate origin via ``coord_origin`` or the page's size in
    # points vs. the page image's size in pixels.
    img_w, img_h = page_img.size
    page_size = getattr(page, "size", None) if page is not None else None
    page_w = float(getattr(page_size, "width", img_w) or img_w)
    page_h = float(getattr(page_size, "height", img_h) or img_h)

    scale_x = img_w / page_w if page_w else 1.0
    scale_y = img_h / page_h if page_h else 1.0

    # Probe bbox origin. Docling uses ``CoordOrigin.BOTTOMLEFT`` for
    # PDF-native bboxes — flip the Y axis when that's the case.
    origin = getattr(bbox, "coord_origin", None)
    origin_str = str(getattr(origin, "value", origin) or "").upper()

    if "BOTTOM" in origin_str:
        # PDF-style: y=0 at page bottom. Convert to top-left origin.
        px_l = l * scale_x
        px_r = r * scale_x
        px_t = (page_h - t) * scale_y
        px_b = (page_h - b) * scale_y
    else:
        px_l = l * scale_x
        px_t = t * scale_y
        px_r = r * scale_x
        px_b = b * scale_y

    # PIL's crop expects (left, upper, right, lower) with upper < lower.
    x0, x1 = sorted((px_l, px_r))
    y0, y1 = sorted((px_t, px_b))

    # Small guard against zero-size crops (rounding errors on very
    # thin formulas). Pad 1 px each side.
    if x1 - x0 < 2:
        x1 = x0 + 2
    if y1 - y0 < 2:
        y1 = y0 + 2

    return page_img.crop((int(x0), int(y0), int(x1), int(y1)))
