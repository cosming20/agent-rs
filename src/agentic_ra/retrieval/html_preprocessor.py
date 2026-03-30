"""Pre-process legislatie.just.ro HTML into semantic HTML for Docling.

legislatie.just.ro renders legal content entirely inside <span> elements
with domain-specific CSS classes (S_ART, S_CAP, S_SEC, etc.) and
``display:block`` styling.  Docling's HTML backend only recognises standard
block-level tags (h1-h6, p, table, ul, …) so all the legal text is treated
as inline content and effectively lost.

This module detects that HTML format and rewrites the ``<span class="S_*">``
elements into semantic HTML that Docling's ``_BLOCK_TAGS`` set recognises,
preserving the legal hierarchy as headings + paragraphs.

Mapping:
    S_CAP  (Capitol)   → <section> with <h2> for title/denomination
    S_SEC  (Secțiune)  → <section> with <h3>
    S_ART  (Articol)   → <section> with <h4>
    S_ALN  (Alineat)   → <p>
    S_LIT  (Litera)    → <p>
    S_LIN  (Linie)     → <p>
    S_PAR  (Paragraf)  → <p>
    S_PUB  (Publicare) → <p>
    S_EMT  (Emitent)   → <p>
    S_SMN  (Semnătură) → <p>
    S_HDR  (Header)    → <h1>
"""

from __future__ import annotations

import logging
import re
from typing import Final

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# Fingerprint: presence of S_ART or S_CAP classes identifies
# legislatie.just.ro HTML.
_LEGI_FINGERPRINT_RE: Final = re.compile(r'class="S_(ART|CAP|PAR)\b')

# Map S_* base classes → target HTML tag for the container element.
_CONTAINER_MAP: Final[dict[str, str]] = {
    "S_CAP": "section",
    "S_SEC": "section",
    "S_ART": "section",
}

# Map S_* base classes → heading level for their _TTL/_DEN children.
_HEADING_MAP: Final[dict[str, str]] = {
    "S_HDR": "h1",
    "S_CAP": "h2",
    "S_SEC": "h3",
    "S_ART": "h4",
}

# All S_* classes whose body content should become <p> elements.
_PARAGRAPH_CLASSES: Final = frozenset({
    "S_ALN",
    "S_LIT",
    "S_LIN",
    "S_PAR",
    "S_PUB",
    "S_EMT",
    "S_SMN",
})


def is_legi_html(html: str) -> bool:
    """Detect whether *html* is legislatie.just.ro legal content.

    Scans the first 50 KB — the site's navigation boilerplate pushes
    the first S_* class to ~18-30 KB depending on document size.
    """
    return bool(_LEGI_FINGERPRINT_RE.search(html[:50_000]))


def preprocess_legi_html(html: str) -> str:
    """Rewrite legislatie.just.ro ``<span class="S_*">`` DOM into semantic HTML.

    The returned HTML keeps ``<head>`` intact (title, charset) and replaces
    the ``<body>`` content with clean semantic elements that Docling's HTML
    backend can parse into a proper DoclingDocument.

    Steps:
        1. Strip ``<script>``, ``<style>``, ``<noscript>`` and hidden elements.
        2. Remove navigation / boilerplate (everything before the first S_* element).
        3. Remove footer boilerplate (everything after the legal content area).
        4. Rewrite S_* spans into semantic HTML.

    Args:
        html: Raw HTML string from legislatie.just.ro.

    Returns:
        Cleaned HTML string with semantic block elements.
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1. Strip scripts, styles, noscript, hidden elements
    for tag in soup(["script", "noscript", "style"]):
        tag.decompose()
    for tag in soup.find_all(attrs={"hidden": True}):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none")):
        tag.decompose()

    # 2. Find the legal content boundary — first S_* element
    first_legal = soup.find(class_=re.compile(r"^S_(HDR|PUB|EMT|PAR|CAP|ART)"))
    if not first_legal:
        logger.warning("No S_* legal content found — returning HTML as-is")
        return html

    # 3. Build a clean body with only legal content
    _rewrite_legal_spans(soup)

    # 4. Remove navigation boilerplate before legal content
    _strip_boilerplate(soup, first_legal)

    return str(soup)


def _get_base_class(tag: Tag) -> str | None:
    """Extract the S_* base class from a tag (e.g. 'S_ART' from 'S_ART_BDY')."""
    classes = tag.get("class", [])
    if not classes:
        return None
    cls = classes[0] if isinstance(classes, list) else classes
    # Match S_XXX but not S_XXX_TTL, S_XXX_BDY, etc.
    if re.match(r"^S_[A-Z]+$", cls):
        return cls
    return None


def _get_sub_class(tag: Tag) -> str | None:
    """Extract full class like S_ART_TTL, S_CAP_BDY."""
    classes = tag.get("class", [])
    if not classes:
        return None
    return classes[0] if isinstance(classes, list) else classes


def _rewrite_legal_spans(soup: BeautifulSoup) -> None:
    """In-place rewrite of S_* spans into semantic HTML elements.

    Strategy: rename *container* spans to block elements and **unwrap**
    their ``_TTL``, ``_DEN``, ``_BDY`` children so content is promoted
    inline.  This avoids nested ``<p>`` inside ``<p>`` which Docling's
    ``_fix_invalid_paragraph_structure`` would discard.

    Order matters — process children (inner) before parents (outer).
    """
    # Pass 0: Remove _SHORT spans (collapsed preview text — duplicates _BDY)
    for tag in list(soup.find_all(class_=re.compile(r"^S_.*_SHORT$"))):
        if isinstance(tag, Tag):
            tag.decompose()

    # Pass 1: Convert heading _TTL/_DEN of structural classes to h-tags.
    #   <span class="S_CAP_TTL">Capitolul I</span>  →  <h2>Capitolul I</h2>
    for tag in list(soup.find_all(class_=re.compile(r"^S_(HDR|CAP|SEC|ART)_(TTL|DEN)$"))):
        if not isinstance(tag, Tag):
            continue
        cls = _get_sub_class(tag)
        if not cls:
            continue
        base = cls.rsplit("_", 1)[0]  # S_CAP_TTL → S_CAP
        heading_tag = _HEADING_MAP.get(base)
        if heading_tag:
            tag.name = heading_tag
            if tag.get("style"):
                del tag["style"]

    # Pass 2: Unwrap _BDY and _TTL of paragraph-level classes.
    #   <span class="S_ALN_BDY">text</span>  →  text  (promoted into parent)
    for tag in list(soup.find_all(class_=re.compile(
        r"^S_(ALN|LIT|LIN|PAR|PUB|EMT|SMN)_(BDY|TTL)$"
    ))):
        if isinstance(tag, Tag):
            tag.unwrap()

    # Pass 3: Convert paragraph-level containers to <p>.
    #   <span class="S_ALN">text</span>  →  <p>text</p>
    for tag in list(soup.find_all(class_=re.compile(
        r"^S_(ALN|LIT|LIN|PAR|PUB|EMT|SMN)$"
    ))):
        if not isinstance(tag, Tag):
            continue
        tag.name = "p"
        if tag.get("style"):
            del tag["style"]

    # Pass 4: Unwrap _BDY of structural classes to <div> (keeps content
    #   inside the parent <section>, avoids nesting issues).
    for tag in list(soup.find_all(class_=re.compile(r"^S_(CAP|SEC|ART)_BDY$"))):
        if isinstance(tag, Tag):
            tag.name = "div"
            if tag.get("style"):
                del tag["style"]

    # Pass 5: Convert structural containers to <section>.
    #   <span class="S_ART">…</span>  →  <section>…</section>
    for tag in list(soup.find_all(class_=re.compile(r"^S_(CAP|SEC|ART)$"))):
        if not isinstance(tag, Tag):
            continue
        tag.name = "section"
        if tag.get("style"):
            del tag["style"]

    # Pass 6: Remove any remaining inline S_LGI reference spans —
    #   keep their text but drop the tag wrapper.
    for tag in list(soup.find_all(class_=re.compile(r"^S_LGI$"))):
        if isinstance(tag, Tag):
            tag.unwrap()


def _strip_boilerplate(soup: BeautifulSoup, first_legal: Tag) -> None:
    """Remove navigation and footer boilerplate from the soup.

    Removes siblings of the legal content's ancestor that appear before it
    (nav menus, site headers) and common footer patterns after it.
    """
    body = soup.find("body")
    if not body or not isinstance(body, Tag):
        return

    # Find the top-level container that holds the legal content.
    # Walk up from first_legal to find a direct child of <body>.
    content_ancestor = first_legal
    while content_ancestor.parent and content_ancestor.parent != body:
        content_ancestor = content_ancestor.parent

    # Remove sibling elements before the legal content (nav, header, etc.)
    # But be careful not to remove the content_ancestor itself.
    for sibling in list(body.children):
        if sibling is content_ancestor:
            break
        if isinstance(sibling, Tag) and sibling.name not in ("head", "meta", "title"):
            # Check if this sibling is a nav/header/boilerplate element
            tag_classes = " ".join(sibling.get("class", []))
            tag_id = sibling.get("id", "")
            if any(kw in str(tag_classes).lower() + str(tag_id).lower()
                   for kw in ("nav", "header", "menu", "slide", "sidebar", "toolbar")):
                sibling.decompose()
            elif sibling.name in ("nav", "header"):
                sibling.decompose()

    # Remove footer elements after legal content
    found_content = False
    for sibling in list(body.children):
        if sibling is content_ancestor:
            found_content = True
            continue
        if found_content and isinstance(sibling, Tag):
            tag_classes = " ".join(sibling.get("class", []))
            tag_id = sibling.get("id", "")
            if any(kw in str(tag_classes).lower() + str(tag_id).lower()
                   for kw in ("footer", "copyright")):
                sibling.decompose()
