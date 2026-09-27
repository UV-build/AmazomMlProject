"""
Amazon ML Challenge 2026 - Business Entity Resolution
normalize.py

Pure, deterministic, stateless normalization functions for business names
and addresses.  Every function is a str → str (or str → list[str]) transform
with no side effects and no external data dependencies.

Grounded in EDA observations:
  - 37 % of matched pairs have differing legal suffixes
  - 33 % have US-state abbreviation vs full-state-name differences
  - 22 % have differing punctuation
  -  9 % have word-order changes
  - DBA / FKA / F/K/A prefixes appear regularly
  - Domain names (.com) used as business names
  - Transliterated names (Devanagari, Tamil, Telugu) appear in Indian entries
  - French entries appear in test set (accented Latin characters)
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

import jellyfish

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _is_latin(text: str) -> bool:
    """Return True if the string contains at least one Latin letter."""
    for ch in text:
        if unicodedata.category(ch).startswith('L'):
            if 'LATIN' in unicodedata.name(ch, ''):
                return True
    return False


def _safe_str(value) -> str:
    """Coerce to str; return '' for None / NaN / 'nan'."""
    if value is None:
        return ''
    s = str(value).strip()
    return '' if s.lower() == 'nan' else s


def collapse_whitespace(text: str) -> str:
    """Replace any run of whitespace (including \t, \n) with a single space."""
    return re.sub(r'\s+', ' ', text).strip()


def remove_accents(text: str) -> str:
    """
    Decompose accented characters to their base + combining form, then
    drop the combining marks.  Preserves non-Latin scripts unchanged.
    Only applies NFC → NFD decomposition on Latin-range characters so that
    Devanagari / Tamil etc. are not distorted.
    """
    # We decompose character by character to leave non-Latin scripts intact.
    out = []
    for ch in text:
        name = unicodedata.name(ch, '')
        if 'LATIN' in name or unicodedata.category(ch) == 'Ll':
            nfd = unicodedata.normalize('NFD', ch)
            out.append(''.join(c for c in nfd if unicodedata.category(c) != 'Mn'))
        else:
            out.append(ch)
    return ''.join(out)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BUSINESS NAME NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

# ── 2a. Legal suffix table (built from EDA observations) ──────────────────────
#
# Each tuple: (regex_pattern, canonical_token)
# Sorted longest-first so that "private limited" is matched before "limited".
# Patterns are applied to the lowercased name.

_LEGAL_SUFFIX_RULES: List[Tuple[str, str]] = [
    # Multi-word Indian forms
    (r'\bprivate\s+limited\b',          'pvtltd'),
    (r'\bpvt\.?\s+ltd\.?\b',            'pvtltd'),
    (r'\bpvt\.?\s+limited\b',           'pvtltd'),
    (r'\bprivate\s+ltd\.?\b',           'pvtltd'),
    # Single-word forms
    (r'\bl\.{0,1}l\.{0,1}c\.?\b',      'llc'),   # LLC / L.L.C.
    (r'\bllc\b',                         'llc'),
    (r'\bl\.{0,1}l\.{0,1}p\.?\b',      'llp'),   # LLP / L.L.P.
    (r'\bllp\b',                         'llp'),
    (r'\bincorporated\b',               'inc'),
    (r'\binc\.?\b',                      'inc'),
    (r'\bcorporation\b',                'corp'),
    (r'\bcorp\.?\b',                     'corp'),
    (r'\blimited\b',                    'ltd'),
    (r'\bltd\.?\b',                     'ltd'),
    (r'\bpty\.?\b',                     'pty'),
    (r'\bplc\.?\b',                     'plc'),
    (r'\bcompany\b',                    'co'),
    (r'\bco\.?\b',                      'co'),
    (r'\bgmbh\b',                       'gmbh'),
    (r'\bsarl\b',                       'sarl'),
    (r'\bs\.{0,1}a\.{0,1}s\.?\b',      'sas'),   # SAS / S.A.S.
    (r'\bs\.{0,1}a\.?\b',              'sa'),    # SA / S.A.
    (r'\bprivate\b',                    'pvt'),
    (r'\bpvt\.?\b',                     'pvt'),
]

# Pre-compile for speed
_LEGAL_COMPILED = [(re.compile(pat, re.IGNORECASE), canon)
                   for pat, canon in _LEGAL_SUFFIX_RULES]

# Regex that matches ANY legal token (for stripping)
_LEGAL_STRIP_RE = re.compile(
    r'\b(?:private\s+limited|pvt\.?\s+ltd\.?|pvt\.?\s+limited|private\s+ltd\.?'
    r'|l\.{0,1}l\.{0,1}c\.?|llc|l\.{0,1}l\.{0,1}p\.?|llp'
    r'|incorporated|inc\.?|corporation|corp\.?|limited|ltd\.?'
    r'|pty\.?|plc\.?|company|co\.?|gmbh|sarl|s\.{0,1}a\.{0,1}s\.?|s\.{0,1}a\.?'
    r'|private|pvt\.?)\b',
    re.IGNORECASE
)

# ── 2b. Noise prefix/suffix table ─────────────────────────────────────────────
# DBA, FKA, AKA, domain names, phone numbers, leading symbol noise

_NOISE_PREFIX_RE = re.compile(
    r'^(?:[-=<>*#@!~^|]+\s*)'           # leading symbol garbage (>> -- == etc.)
    r'|(?:\b(?:dba|d\.b\.a\.?|fka|f\.k\.a\.?|aka|formerly\s+known\s+as'
    r'|formerly|doing\s+business\s+as)\b[\s,:-]*)',
    re.IGNORECASE
)

_PHONE_RE   = re.compile(r'\b[\d]{7,15}\b')          # appended phone numbers
_DOMAIN_RE  = re.compile(r'\b\S+\.(com|net|org|in|co\.in|biz|info)\b', re.IGNORECASE)
_URL_RE     = re.compile(r'https?://\S+', re.IGNORECASE)

# ── 2c. & / and normalisation ─────────────────────────────────────────────────
_AMP_AND_RE = re.compile(r'\s*&\s*', re.IGNORECASE)   # "&" → " and "


def normalize_punctuation_name(text: str) -> str:
    """
    Normalise punctuation in a business name:
      - Replace '&' with ' and '
      - Drop brackets []()
      - Replace hyphens / slashes between word chars with a space
      - Drop remaining punctuation chars that carry no semantic meaning
      - Collapse whitespace
    """
    text = _AMP_AND_RE.sub(' and ', text)
    text = re.sub(r'[(){}\[\]]', ' ', text)                 # brackets
    text = re.sub(r'(?<=\w)[/\\-](?=\w)', ' ', text)        # word-hyphen/slash-word
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)  # all other punctuation
    return collapse_whitespace(text)


def strip_legal_suffixes(text: str) -> str:
    """
    Remove all recognised legal-suffix tokens from the text.
    Handles both original forms (e.g. 'Private Limited', 'LLC') and the
    canonical forms produced by canonicalize_legal_suffixes (e.g. 'pvtltd').
    """
    # Strip canonical tokens first (output of canonicalize_legal_suffixes)
    _CANONICAL_STRIP_RE = re.compile(
        r'\b(?:pvtltd|llc|llp|inc|corp|ltd|pvt|pty|plc|co|gmbh|sarl|sas|sa)\b',
        re.IGNORECASE
    )
    text = _CANONICAL_STRIP_RE.sub(' ', text)
    # Then strip any residual original forms
    text = _LEGAL_STRIP_RE.sub(' ', text)
    return collapse_whitespace(text)


def canonicalize_legal_suffixes(text: str) -> str:
    """
    Replace each legal-suffix variant with its canonical short form.
    E.g. 'Private Limited' → 'pvtltd', 'Incorporated' → 'inc'.
    Applied after lowercasing.
    """
    for regex, canon in _LEGAL_COMPILED:
        text = regex.sub(' ' + canon + ' ', text)
    return collapse_whitespace(text)


def strip_noise_prefixes(text: str) -> str:
    """Iteratively strip DBA / FKA / leading-symbol noise from the start."""
    prev = None
    while prev != text:
        prev = text
        text = _NOISE_PREFIX_RE.sub('', text).strip()
    return text


def strip_domain_and_phone(text: str) -> str:
    """Remove .com domain strings and appended phone numbers."""
    text = _URL_RE.sub(' ', text)
    text = _DOMAIN_RE.sub(' ', text)
    text = _PHONE_RE.sub(' ', text)
    return collapse_whitespace(text)


# ── 2d. Top-level name representations ────────────────────────────────────────

def normalize_name(raw: str) -> str:
    """
    Canonical lowercased name.  Steps:
      1. Safe-string coerce
      2. Strip DBA/FKA noise prefixes
      3. Strip domain names and phone numbers
      4. Remove accents (Latin only, preserves Devanagari/Tamil/Telugu)
      5. Lowercase
      6. Canonicalize legal suffixes BEFORE punctuation stripping so that
         dotted forms like 'L.L.C.' are matched intact (e.g. L.L.C. → llc)
      7. Normalise punctuation (& → and, drop brackets, hyphens→space)
      8. Collapse whitespace
    """
    text = _safe_str(raw)
    if not text:
        return ''
    text = strip_noise_prefixes(text)
    text = strip_domain_and_phone(text)
    text = remove_accents(text)
    text = text.lower()
    text = canonicalize_legal_suffixes(text)   # ← before punct stripping
    text = normalize_punctuation_name(text)
    return collapse_whitespace(text)


def core_name(raw: str) -> str:
    """
    Core name: canonical name with all legal suffixes stripped.
    Use for similarity comparisons where suffix noise should be ignored.
    """
    return strip_legal_suffixes(normalize_name(raw))


def sorted_tokens(raw: str) -> str:
    """
    Sorted-token representation (token bag).
    Handles word-order changes: 'AP Inc Hospitality' and 'AP Hospitality Inc'
    both map to the same string.
    """
    tokens = core_name(raw).split()
    return ' '.join(sorted(tokens))


def name_tokens(raw: str) -> List[str]:
    """Return the token list of the core name."""
    return core_name(raw).split()


def phonetic_name(raw: str) -> str:
    """
    Phonetic representation using Metaphone via jellyfish.
    Applied per token on the core name; non-ASCII tokens are kept as-is
    (transliterated names cannot be phonetically encoded with English Metaphone).
    Tokens are joined with a space.
    """
    tokens = name_tokens(raw)
    result = []
    for tok in tokens:
        if tok.isascii():
            try:
                code = jellyfish.metaphone(tok)
                result.append(code if code else tok)
            except Exception:
                result.append(tok)
        else:
            result.append(tok)   # preserve non-Latin tokens verbatim
    return ' '.join(result)


def soundex_name(raw: str) -> str:
    """
    Soundex per token (ASCII tokens only).  More coarse than Metaphone;
    useful as a secondary blocking / grouping key.
    """
    tokens = name_tokens(raw)
    result = []
    for tok in tokens:
        if tok.isascii() and len(tok) >= 2:
            try:
                result.append(jellyfish.soundex(tok))
            except Exception:
                result.append(tok)
        else:
            result.append(tok)
    return ' '.join(result)


def jaro_name_repr(raw: str) -> str:
    """
    Returns the normalized canonical name (same as normalize_name).
    Jaro-Winkler similarity is applied at comparison time (in features.py);
    this function just returns the cleaned string to compare against.
    """
    return normalize_name(raw)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ADDRESS NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

# ── 3a. US state table (abbreviation ↔ full name) ─────────────────────────────
# Built from EDA: 2.57 M pairs had state-abbr vs full-name mismatch.

_US_STATE_ABBR: dict[str, str] = {
    'AL':'alabama','AK':'alaska','AZ':'arizona','AR':'arkansas','CA':'california',
    'CO':'colorado','CT':'connecticut','DE':'delaware','FL':'florida','GA':'georgia',
    'HI':'hawaii','ID':'idaho','IL':'illinois','IN':'indiana','IA':'iowa',
    'KS':'kansas','KY':'kentucky','LA':'louisiana','ME':'maine','MD':'maryland',
    'MA':'massachusetts','MI':'michigan','MN':'minnesota','MS':'mississippi',
    'MO':'missouri','MT':'montana','NE':'nebraska','NV':'nevada','NH':'new hampshire',
    'NJ':'new jersey','NM':'new mexico','NY':'new york','NC':'north carolina',
    'ND':'north dakota','OH':'ohio','OK':'oklahoma','OR':'oregon','PA':'pennsylvania',
    'RI':'rhode island','SC':'south carolina','SD':'south dakota','TN':'tennessee',
    'TX':'texas','UT':'utah','VT':'vermont','VA':'virginia','WA':'washington',
    'WV':'west virginia','WI':'wisconsin','WY':'wyoming','DC':'district of columbia',
}
_US_STATE_FULL_TO_ABBR: dict[str, str] = {v: k for k, v in _US_STATE_ABBR.items()}

# ── 3b. Indian state/UT abbreviations (observed in dataset) ───────────────────
_IN_STATE_ABBR: dict[str, str] = {
    'DL':'delhi','HR':'haryana','MH':'maharashtra','KA':'karnataka',
    'TN':'tamil nadu','TG':'telangana','AP':'andhra pradesh','UP':'uttar pradesh',
    'RJ':'rajasthan','GJ':'gujarat','MP':'madhya pradesh','WB':'west bengal',
    'PB':'punjab','KL':'kerala','OR':'odisha','CG':'chhattisgarh',
    'JK':'jammu and kashmir','HP':'himachal pradesh','BR':'bihar',
    'JH':'jharkhand','AS':'assam','UK':'uttarakhand',
}
_IN_STATE_FULL_TO_ABBR: dict[str, str] = {v: k for k, v in _IN_STATE_ABBR.items()}

# ── 3c. French department/region abbreviations (for test set) ─────────────────
_FR_REGION_MAP: dict[str, str] = {
    'IDF':'ile-de-france', 'PACA':'provence-alpes-cote d azur',
    'ARA':'auvergne-rhone-alpes', 'BFC':'bourgogne-franche-comte',
    'BRE':'bretagne', 'CVL':'centre-val de loire', 'COR':'corse',
    'GES':'grand est', 'HDF':'hauts-de-france', 'NOR':'normandie',
    'NAQ':'nouvelle-aquitaine', 'OCC':'occitanie', 'PDL':'pays de la loire',
}

# ── 3d. Street type abbreviation table ────────────────────────────────────────
# EDA showed St/Street, Ave/Avenue, Blvd/Boulevard, Rd/Road, Dr/Drive, etc.
_STREET_ABBR: dict[str, str] = {
    r'\bstreet\b':    'st',
    r'\bavenue\b':    'ave',
    r'\bboulevard\b': 'blvd',
    r'\bdrive\b':     'dr',
    r'\broad\b':      'rd',
    r'\blane\b':      'ln',
    r'\bcourt\b':     'ct',
    r'\bplace\b':     'pl',
    r'\bparkway\b':   'pkwy',
    r'\bhighway\b':   'hwy',
    r'\bsquare\b':    'sq',
    r'\bterrace\b':   'ter',
    r'\bcircle\b':    'cir',
    r'\btrail\b':     'trl',
    r'\bway\b':       'way',   # kept as-is; included for symmetry
    r'\bexpress(?:way)?\b': 'expy',
    r'\bnorth\b':     'n',
    r'\bsouth\b':     's',
    r'\beast\b':      'e',
    r'\bwest\b':      'w',
    r'\bnortheast\b': 'ne', r'\bnorthwest\b': 'nw',
    r'\bsoutheast\b': 'se', r'\bsouthwest\b': 'sw',
    r'\bapartment\b': 'apt',
    r'\bsuite\b':     'ste',
    r'\bbuilding\b':  'bldg',
    r'\bfloor\b':     'fl',
}
_STREET_COMPILED = [(re.compile(pat, re.IGNORECASE), abbr)
                    for pat, abbr in _STREET_ABBR.items()]

# ── 3e. Patterns for extraction ───────────────────────────────────────────────

# US ZIP: 5-digit or 5+4
_US_ZIP_RE   = re.compile(r'\b(\d{5})(?:-\d{4})?\b')
# India PIN: exactly 6 digits
_IN_PIN_RE   = re.compile(r'\b([1-9]\d{5})\b')
# French postal: exactly 5 digits starting 0-9 (not preceded by a letter)
_FR_POST_RE  = re.compile(r'(?<!\w)(\d{5})(?!\d)')

# PO Box, PMB
_POBOX_RE    = re.compile(r'\b(?:p\.?o\.?\s*box|pmb|po\s+box)\s*[\d#-]*',re.IGNORECASE)

# Unit / Suite / Apt markers — strip from address tokens for city extraction
_UNIT_RE     = re.compile(
    r'\b(?:unit|apt\.?|apartment|suite|ste\.?|#|no\.?|door\s*no\.?|flat\s*no\.?'
    r'|shop\s*no\.?|plot\s*no\.?|h\.?no\.?|house\s*no\.?)\s*[\w/,-]*',
    re.IGNORECASE
)

# State-like token extraction (US: 2-letter uppercase token)
_US_STATE_TOKEN_RE = re.compile(r'\b([A-Z]{2})\b')


def normalize_punctuation_addr(text: str) -> str:
    """
    Normalise punctuation in an address:
      - Keep commas (they separate components) but normalise spacing around them
      - Replace hyphens between digit sequences with a space (address ranges)
      - Replace other non-alphanumeric, non-comma chars with space
      - Collapse whitespace
    """
    text = re.sub(r'\s*,\s*', ', ', text)                   # normalise comma spacing
    text = re.sub(r'(?<=\d)-(?=\d)', ' ', text)             # digit-hyphen-digit → space
    text = re.sub(r'[^\w\s,]', ' ', text, flags=re.UNICODE) # drop remaining symbols
    return collapse_whitespace(text)


def expand_state_abbreviations(text: str, country: str = '') -> str:
    """
    Expand 2-letter US-state abbreviations to their full lowercase name.
    Applied only when country is explicitly identified as US.

    Empty / unknown country strings are NOT treated as US — they fall through
    unchanged so that French or other unseen-country records are not corrupted
    by false US-state expansions (e.g. 'OR' or 'CA' in a French address).
    """
    country_lc = (country or '').lower().strip()
    # Only expand when we are confident the record is US.
    # Blank country is left unchanged — generic normalization is safer.
    if country_lc not in ('us', 'united states', 'usa', 'u.s.', 'u.s.a.'):
        return text

    def _replace(m: re.Match) -> str:
        tok = m.group(1).upper()
        return _US_STATE_ABBR.get(tok, m.group(0))

    return _US_STATE_TOKEN_RE.sub(_replace, text)


def abbreviate_street_types(text: str) -> str:
    """
    Normalise street-type words to their short forms:
    'Street' → 'st', 'Avenue' → 'ave', 'Boulevard' → 'blvd', etc.
    This reduces the impact of abbreviated vs full address differences.
    """
    for regex, abbr in _STREET_COMPILED:
        text = regex.sub(abbr, text)
    return text


def normalize_address(raw: str, country: str = '') -> str:
    """
    Cleaned address string.  Steps:
      1. Safe-string coerce
      2. Remove accents (Latin only)
      3. Lowercase
      4. Normalise punctuation
      5. Expand state abbreviations (US)
      6. Abbreviate street types
      7. Collapse whitespace
    """
    text = _safe_str(raw)
    if not text:
        return ''
    text = remove_accents(text)
    text = text.lower()
    text = normalize_punctuation_addr(text)
    text = expand_state_abbreviations(text, country)
    text = abbreviate_street_types(text)
    return collapse_whitespace(text)


def address_tokens(raw: str, country: str = '') -> List[str]:
    """Return the token list of the normalised address."""
    return normalize_address(raw, country).split()


def sorted_address_tokens(raw: str, country: str = '') -> str:
    """
    Sorted token bag for address — handles component reordering.
    E.g. 'Greensboro, 1604 Washington Street, NC'
      == '1604 Washington Street, Greensboro, NC' after sorting.
    """
    return ' '.join(sorted(address_tokens(raw, country)))


# ── 3f. Field extraction ──────────────────────────────────────────────────────

def extract_postal_code(raw: str, country: str = '') -> Optional[str]:
    """
    Extract the first plausible postal/PIN/ZIP code from the address string.
    Returns the raw digit string, or None if not found.
    Order of precedence:
      - India PIN (6 digits starting 1–9)
      - US ZIP (5 digits)
      - Generic 5-digit (French / other)
    Country is used as a hint but we try all patterns if the first fails.
    """
    text = _safe_str(raw)
    country_lc = (country or '').lower()

    if 'india' in country_lc:
        m = _IN_PIN_RE.search(text)
        if m:
            return m.group(1)

    if country_lc in ('us', 'united states', 'usa', 'u.s.', 'u.s.a.'):
        m = _US_ZIP_RE.search(text)
        if m:
            return m.group(1)

    # Generic: try all
    for pattern in (_IN_PIN_RE, _US_ZIP_RE, _FR_POST_RE):
        m = pattern.search(text)
        if m:
            return m.group(1)

    return None


def extract_state_region(raw: str, country: str = '') -> Optional[str]:
    """
    Extract the most likely state/region token from the address.
    Returns a lowercased canonical form, or None.

    Strategy:
      - For US (explicitly): look for a known 2-letter state code or full name.
      - For India (explicitly): look for known state abbreviations or names.
      - For France (explicitly): try _FR_REGION_MAP abbreviations, then fall
        back to the last non-numeric comma-separated segment.
      - For empty / other / unseen country: last non-numeric comma segment.
        Blank country does NOT run US state detection to avoid corrupting
        French or other unseen-country addresses.
    """
    text = _safe_str(raw)
    if not text:
        return None

    country_lc = (country or '').lower()
    text_lc = text.lower()

    # US — only when country is explicitly US
    if country_lc in ('us', 'united states', 'usa', 'u.s.', 'u.s.a.'):
        # full state name — but only when NOT immediately followed by a
        # street-type word (to avoid "Washington Street" → WA false positive)
        _STREET_SUFFIX_RE = re.compile(
            r'\b(?:street|st|avenue|ave|boulevard|blvd|drive|dr|road|rd|'
            r'lane|ln|court|ct|place|pl|parkway|pkwy|highway|hwy|'
            r'way|circle|cir|trail|trl|terrace|ter|square|sq)\b',
            re.IGNORECASE
        )
        for full_name in sorted(_US_STATE_FULL_TO_ABBR, key=len, reverse=True):
            pat = r'\b' + re.escape(full_name) + r'\b'
            for m in re.finditer(pat, text_lc):
                after = text_lc[m.end():].strip()
                if _STREET_SUFFIX_RE.match(after):
                    continue
                return _US_STATE_FULL_TO_ABBR[full_name]
        for part in re.split(r'[,\s]+', text):
            tok_upper = part.strip('.,').upper()
            if tok_upper in _US_STATE_ABBR:
                return tok_upper

    # India
    if 'india' in country_lc:
        for full_name in sorted(_IN_STATE_FULL_TO_ABBR, key=len, reverse=True):
            if re.search(r'\b' + re.escape(full_name) + r'\b', text_lc):
                return _IN_STATE_FULL_TO_ABBR[full_name]
        for tok in text.split():
            tok_upper = tok.strip('.,').upper()
            if tok_upper in _IN_STATE_ABBR:
                return tok_upper

    # France — try the region abbreviation map, then fall through to generic
    if 'france' in country_lc:
        for tok in text.upper().split():
            tok_clean = tok.strip('.,')
            if tok_clean in _FR_REGION_MAP:
                return _FR_REGION_MAP[tok_clean]   # e.g. 'IDF' -> 'ile-de-france'

    # Generic fallback (France without a matched abbreviation, blank country,
    # or any other unseen country): return last non-postal comma segment.
    parts = [p.strip() for p in text.split(',') if p.strip()]
    for part in reversed(parts):
        if not re.fullmatch(r'[\d\s-]+', part):
            return part.lower()

    return None


def extract_city(raw: str, country: str = '') -> Optional[str]:
    """
    Heuristic city extraction from address.
    Strategy:
      - Remove house/unit/plot numbers (leading digit tokens)
      - Remove known state/region tokens
      - Remove postal codes
      - Remove PO Box / PMB
      - Return the most likely city token (penultimate comma segment for US;
        for India often appears after locality details)
    Returns a lowercased string or None.
    """
    text = _safe_str(raw)
    if not text:
        return None

    country_lc = (country or '').lower()

    # Work with comma-separated parts
    parts = [p.strip() for p in text.split(',') if p.strip()]
    if not parts:
        return None

    # Remove parts that are just postal codes
    zip_like = re.compile(r'^\d{5,6}$')
    parts = [p for p in parts if not zip_like.match(p.strip())]

    # For US addresses the pattern is typically:
    #   <street>, <city>, <state> <zip>
    # So city is the second-to-last non-empty part after removing state/zip.
    if country_lc in ('us', 'united states', 'usa', 'u.s.', 'u.s.a.'):
        # Drop the last part if it looks like:
        #   - "STATE ZIP" or just "STATE" (2-letter abbr)
        #   - a full US state name (e.g. "North Carolina", "Texas")
        if parts:
            last_lc = parts[-1].strip().lower()
            last_toks = parts[-1].strip().upper().split()
            is_state_abbr = last_toks and last_toks[0] in _US_STATE_ABBR
            is_state_full = last_lc in _US_STATE_FULL_TO_ABBR
            if is_state_abbr or is_state_full:
                parts = parts[:-1]
        if len(parts) >= 2:
            city_candidate = parts[-1].strip().lower()
            # Strip leading numbers from city candidate
            city_candidate = re.sub(r'^\d+\s*', '', city_candidate).strip()
            return city_candidate if city_candidate else None

    # For India: addresses are less structured; try to find a part that looks
    # like a city (doesn't start with digits, isn't a known state name).
    if 'india' in country_lc:
        state_tok = extract_state_region(raw, country)
        for part in reversed(parts):
            part_lc = part.strip().lower()
            if part_lc and not re.match(r'^\d', part_lc):
                if state_tok and part_lc == state_tok.lower():
                    continue
                # Skip if it's a known state name
                if part_lc in _IN_STATE_FULL_TO_ABBR:
                    continue
                return part_lc
        return None

    # Generic fallback: return middle comma segment
    if len(parts) >= 2:
        return parts[len(parts) // 2].strip().lower()
    return parts[0].strip().lower() if parts else None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — COMBINED RECORD NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field


@dataclass
class NormalizedName:
    raw:            str
    canonical:      str   # lowercased, punct-normalised, legal-suffixes canonicalized
    core:           str   # canonical with all legal suffixes stripped
    sorted_tokens:  str   # sorted token bag of core name
    tokens:         List[str]
    phonetic:       str   # Metaphone per token
    soundex:        str   # Soundex per token


@dataclass
class NormalizedAddress:
    raw:            str
    country:        str
    cleaned:        str        # lowercased, punct-normalised, abbr expanded
    tokens:         List[str]
    sorted_tokens:  str
    postal_code:    Optional[str]
    state_region:   Optional[str]
    city:           Optional[str]


def normalize_record_name(raw: str) -> NormalizedName:
    """Produce all name representations for a single business name."""
    canon  = normalize_name(raw)
    core_  = strip_legal_suffixes(canon)
    toks   = core_.split()
    return NormalizedName(
        raw           = _safe_str(raw),
        canonical     = canon,
        core          = core_,
        sorted_tokens = ' '.join(sorted(toks)),
        tokens        = toks,
        phonetic      = phonetic_name(raw),
        soundex       = soundex_name(raw),
    )


def normalize_record_address(raw: str, country: str = '') -> NormalizedAddress:
    """Produce all address representations for a single address string."""
    cleaned = normalize_address(raw, country)
    toks    = cleaned.split()
    return NormalizedAddress(
        raw           = _safe_str(raw),
        country       = _safe_str(country),
        cleaned       = cleaned,
        tokens        = toks,
        sorted_tokens = ' '.join(sorted(toks)),
        postal_code   = extract_postal_code(raw, country),
        state_region  = extract_state_region(raw, country),
        city          = extract_city(raw, country),
    )
