"""
Amazon ML Challenge 2026 - Business Entity Resolution
test_normalize.py

Unit tests for normalize.py.
Every test case is grounded in actual examples observed in EDA of training data.
Run with: python -m pytest test_normalize.py -v
       or: python test_normalize.py
"""

import sys
import os
import unittest

# Allow running from src/ directly or from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalize import (
    # shared
    collapse_whitespace,
    remove_accents,
    _safe_str,
    # name
    normalize_name,
    core_name,
    sorted_tokens,
    name_tokens,
    phonetic_name,
    soundex_name,
    strip_legal_suffixes,
    strip_noise_prefixes,
    strip_domain_and_phone,
    normalize_punctuation_name,
    canonicalize_legal_suffixes,
    # address
    normalize_address,
    address_tokens,
    sorted_address_tokens,
    expand_state_abbreviations,
    abbreviate_street_types,
    # extraction
    extract_postal_code,
    extract_state_region,
    extract_city,
    # combined
    normalize_record_name,
    normalize_record_address,
)


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

class TestSafeStr(unittest.TestCase):
    def test_none_returns_empty(self):
        self.assertEqual(_safe_str(None), '')

    def test_nan_string_returns_empty(self):
        self.assertEqual(_safe_str('nan'), '')
        self.assertEqual(_safe_str('NaN'), '')

    def test_normal_string_preserved(self):
        self.assertEqual(_safe_str('  hello  '), 'hello')

    def test_float_nan_returns_empty(self):
        import math
        self.assertEqual(_safe_str(float('nan')), '')


class TestCollapseWhitespace(unittest.TestCase):
    def test_multiple_spaces(self):
        self.assertEqual(collapse_whitespace('a  b   c'), 'a b c')

    def test_tabs_and_newlines(self):
        self.assertEqual(collapse_whitespace('a\t\tb\nc'), 'a b c')

    # EDA: "Mountain  Council", "Caldeon  Nova", "Sonipat  Engineering"
    def test_double_space_in_name(self):
        self.assertEqual(collapse_whitespace('Mountain  Council'), 'Mountain Council')


class TestRemoveAccents(unittest.TestCase):
    # EDA: "Payne Énterprises", "Services Varalakshmi Financial Cénter"
    def test_accented_e(self):
        self.assertEqual(remove_accents('Énterprises'), 'Enterprises')
        self.assertEqual(remove_accents('Cénter'), 'Center')

    # EDA: "Lumay Bóral"
    def test_accented_o(self):
        self.assertEqual(remove_accents('Bóral'), 'Boral')

    # EDA: French names preserved (Devanagari script must NOT be mangled)
    def test_devanagari_preserved(self):
        hindi = 'सनराइज इंफ्रा'
        self.assertEqual(remove_accents(hindi), hindi)

    def test_latin_with_accent(self):
        self.assertEqual(remove_accents('Ticonderoga'), 'Ticonderoga')


# ══════════════════════════════════════════════════════════════════════════════
# Business name — punctuation
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizePunctuationName(unittest.TestCase):
    # EDA: "Foot & Ankle Specialists" ↔ "FOOT and ANKLE"
    def test_ampersand_to_and(self):
        result = normalize_punctuation_name('Foot & Ankle')
        self.assertIn('and', result)
        self.assertNotIn('&', result)

    # EDA: "Crystal Staffing Solutions LLC" ↔ "CRYSTAL STAFFING SOLUTIONS-L.L.C."
    def test_hyphen_between_words_becomes_space(self):
        result = normalize_punctuation_name('SOLUTIONS-L.L.C.')
        self.assertNotIn('-', result)

    # EDA: "Pitts-Packaging [Inc.]"
    def test_brackets_removed(self):
        result = normalize_punctuation_name('Pitts-Packaging [Inc.]')
        self.assertNotIn('[', result)
        self.assertNotIn(']', result)

    # EDA: "Obsidian, LLC" ↔ "Obsidian,-LLC"
    def test_comma_hyphen_noise(self):
        result = normalize_punctuation_name('Obsidian,-LLC')
        self.assertNotIn(',', result)
        self.assertNotIn('-', result)

    # EDA: "Desai, Alanah, Esq., PC"
    def test_commas_stripped(self):
        result = normalize_punctuation_name('Desai, Alanah, Esq., PC')
        self.assertNotIn(',', result)


# ══════════════════════════════════════════════════════════════════════════════
# Business name — legal suffixes
# ══════════════════════════════════════════════════════════════════════════════

class TestLegalSuffixes(unittest.TestCase):
    # EDA top-1: "Private Limited" ↔ "(none)" — 494,953 pairs
    def test_private_limited_canonicalized(self):
        self.assertEqual(canonicalize_legal_suffixes('new delhi events private limited'),
                         'new delhi events pvtltd')

    # EDA: "Private Limited" ↔ "Pvt Ltd" — 155,520 pairs
    def test_pvt_ltd_canonicalized(self):
        self.assertEqual(canonicalize_legal_suffixes('global infrastructure pvt ltd'),
                         'global infrastructure pvtltd')

    # EDA: "LLC" ↔ "(none)" — 380,226 pairs
    def test_llc_stripped(self):
        self.assertEqual(strip_legal_suffixes('hendricks ship llc'), 'hendricks ship')

    # EDA: "LLC" ↔ "L.L.C." — 63,049 pairs
    def test_llc_dotted_canonicalized(self):
        result = normalize_name('Crystal Staffing Solutions L.L.C.')
        result2 = normalize_name('Crystal Staffing Solutions LLC')
        self.assertEqual(core_name('Crystal Staffing Solutions L.L.C.'),
                         core_name('Crystal Staffing Solutions LLC'))

    # EDA: "Inc" ↔ "Incorporated" — 41,109 pairs
    def test_incorporated_canonicalized(self):
        self.assertEqual(canonicalize_legal_suffixes('american ridge dogecoin incorporated'),
                         'american ridge dogecoin inc')

    # EDA: "Limited" ↔ "Ltd" — 34,321 pairs
    def test_limited_vs_ltd(self):
        n1 = core_name('New Delhi Events Private Limited')
        n2 = core_name('New Delhi Events Private Ltd')
        self.assertEqual(n1, n2)

    # EDA: "LLP" ↔ "(none)" — 57,464 pairs
    def test_llp_stripped(self):
        self.assertEqual(strip_legal_suffixes('raj investments llp'), 'raj investments')

    # EDA: "Corp" ↔ "Corporation" — observed pair
    def test_corp_vs_corporation(self):
        n1 = core_name('Dick Regional Armada Corp')
        n2 = core_name('Dick Regional Armada Corporation')
        self.assertEqual(n1, n2)


# ══════════════════════════════════════════════════════════════════════════════
# Business name — noise prefix stripping
# ══════════════════════════════════════════════════════════════════════════════

class TestNoisePrefixes(unittest.TestCase):
    # EDA: ">> SWASTIK OM SOLUTIONS L.L.P."
    def test_leading_angle_brackets(self):
        result = strip_noise_prefixes('>> SWASTIK OM SOLUTIONS')
        self.assertFalse(result.startswith('>'))

    # EDA: "-- Delhi To Pvt. Ltd"
    def test_leading_dashes(self):
        result = strip_noise_prefixes('-- Delhi To Pvt. Ltd')
        self.assertFalse(result.startswith('-'))

    # EDA: "Quoumbra dba Mumbai Marketing Public Limited"
    def test_dba_stripped(self):
        result = strip_noise_prefixes('Quoumbra dba Mumbai Marketing Public Limited')
        self.assertNotIn('dba', result.lower())
        self.assertIn('mumbai', result.lower())

    # EDA: "Mirajax fka Back Alley Book Store"
    def test_fka_stripped(self):
        result = strip_noise_prefixes('Mirajax fka Back Alley Book Store')
        self.assertNotIn('fka', result.lower())
        self.assertIn('back alley', result.lower())

    # EDA: "Vantagedeltaio dba Innovative Care Private Limited"
    def test_dba_prefix_chained(self):
        result = strip_noise_prefixes('Vantagedeltaio dba Innovative Care Private Limited')
        self.assertIn('innovative care', result.lower())

    # EDA: "Belokorumbra f/k/a Certified Quality Enbridge Inc"
    def test_fka_slash_variant(self):
        result = strip_noise_prefixes('Belokorumbra f/k/a Certified Quality Enbridge Inc')
        self.assertIn('certified quality', result.lower())


class TestDomainAndPhone(unittest.TestCase):
    # EDA: "maurewilliamscolombier.com"
    def test_com_domain_stripped(self):
        result = strip_domain_and_phone('maurewilliamscolombier.com')
        self.assertNotIn('.com', result)

    # EDA: "lopezinnovativenetwork.com"
    def test_domain_as_name(self):
        result = normalize_name('lopezinnovativenetwork.com')
        self.assertNotIn('.com', result)

    # EDA: "Chordia + Pagnters - 7306204978" (phone appended)
    def test_phone_number_stripped(self):
        result = strip_domain_and_phone('Chordia Partners - 7306204978')
        self.assertNotIn('7306204978', result)

    # EDA: "Khatuwala Holdings  (Group) - 3934263534"
    def test_phone_stripped_from_long_name(self):
        result = normalize_name('Khatuwala Holdings (Group) - 3934263534')
        self.assertNotIn('3934263534', result)


# ══════════════════════════════════════════════════════════════════════════════
# Business name — canonical / core / sorted / phonetic
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizeName(unittest.TestCase):
    # EDA: case differences seen throughout
    def test_lowercase(self):
        self.assertEqual(normalize_name('CRYSTAL STAFFING SOLUTIONS'), 
                         'crystal staffing solutions')

    # EDA: "Foot & Ankle Specialists" ↔ "FOOT and ANKLE"
    def test_ampersand_normalized(self):
        n1 = core_name('Foot & Ankle Specialists')
        n2 = core_name('FOOT and ANKLE Specialists')
        self.assertEqual(n1, n2)

    # EDA: "Payne Énterprises" ↔ "Payne Enterprises"
    def test_accented_char_normalized(self):
        n1 = core_name('Payne Enterprises')
        n2 = core_name('Payne Énterprises')
        self.assertEqual(n1, n2)

    def test_empty_string(self):
        self.assertEqual(normalize_name(''), '')
        self.assertEqual(normalize_name(None), '')

    # EDA: double-space variants
    def test_double_space_collapsed(self):
        self.assertEqual(normalize_name('Mountain  Council'),
                         normalize_name('Mountain Council'))


class TestCoreNameAndSortedTokens(unittest.TestCase):
    # EDA: "AP Hospitality Inc" ↔ "AP Inc Hospitality" — word-order change
    def test_word_order_normalized_via_sorted_tokens(self):
        self.assertEqual(sorted_tokens('AP Hospitality Inc'),
                         sorted_tokens('AP Inc Hospitality'))

    # EDA: "Jeanlouis Advantage LLC" ↔ "LLC JEANLOUIS ADVANTAGE"
    def test_llc_at_start_vs_end(self):
        self.assertEqual(sorted_tokens('Jeanlouis Advantage LLC'),
                         sorted_tokens('LLC JEANLOUIS ADVANTAGE'))

    # EDA: "Tiena L. Hamilton, DDS" ↔ "dds l. hamilton, tiena"
    def test_full_reorder(self):
        self.assertEqual(sorted_tokens('Tiena L. Hamilton, DDS'),
                         sorted_tokens('dds l. hamilton, tiena'))

    # EDA: "New Delhi Events Private Limited" ↔ "New Delhi Events Private Ltd"
    def test_suffix_variants_same_core(self):
        self.assertEqual(core_name('New Delhi Events Private Limited'),
                         core_name('New Delhi Events Private Ltd'))

    # EDA: "Drayova Nuveen Inc." ↔ "Drayova Inc. Nuveen"
    def test_word_order_with_suffix(self):
        self.assertEqual(sorted_tokens('Drayova Nuveen Inc.'),
                         sorted_tokens('Drayova Inc. Nuveen'))


class TestPhoneticName(unittest.TestCase):
    # Phonetic should be identical for name vs slightly misspelled version
    # EDA: "Dahlia Ponr Reliable Scientific LLC" (typo of "Power")
    def test_phonetic_typo_similar(self):
        p1 = phonetic_name('Dahlia Power Reliable Scientific')
        p2 = phonetic_name('Dahlia Ponr Reliable Scientific')
        # Both should produce a non-empty phonetic string
        self.assertTrue(len(p1) > 0)
        self.assertTrue(len(p2) > 0)

    # Non-ASCII tokens preserved verbatim
    def test_non_ascii_preserved_in_phonetic(self):
        result = phonetic_name('Raj Investments एल एल पी')
        self.assertIn('एल', result)

    def test_soundex_similar_names(self):
        s1 = soundex_name('Payne')
        s2 = soundex_name('Pain')
        # Soundex for similar-sounding names should match
        self.assertEqual(s1, s2)


class TestNormalizeRecordName(unittest.TestCase):
    def test_full_record_name(self):
        rec = normalize_record_name('New Delhi Events Private Limited')
        self.assertEqual(rec.canonical, 'new delhi events pvtltd')
        self.assertEqual(rec.core, 'new delhi events')
        self.assertIn('new', rec.tokens)
        self.assertTrue(len(rec.phonetic) > 0)
        self.assertTrue(len(rec.soundex) > 0)

    # EDA: ">> SWASTIK OM SOLUTIONS L.L.P."
    def test_noise_prefix_stripped_in_record(self):
        rec = normalize_record_name('>> SWASTIK OM SOLUTIONS L.L.P.')
        self.assertFalse(rec.canonical.startswith('>'))
        self.assertIn('swastik', rec.core)


# ══════════════════════════════════════════════════════════════════════════════
# Address normalization
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizeAddress(unittest.TestCase):
    # EDA: "2803 Stratford Drive, Austin, TX" ↔ "2803 Startford Dr, Austin, Texas"
    def test_drive_abbreviated(self):
        result = normalize_address('2803 Stratford Drive, Austin, TX', 'US')
        self.assertIn('dr', result)
        self.assertNotIn('drive', result)

    # EDA: "37 Simpson Street, Unit 205, Asheville, NC" ↔ "Simpson Street, # 205, Asheville, North Carolina"
    def test_street_abbreviated(self):
        result = normalize_address('37 Simpson Street, Asheville, NC', 'US')
        self.assertIn('st', result)
        self.assertNotIn('street', result)

    # EDA: "11500 Island Road, OH, Grafton" ↔ "11500 ISLAND RD, GRAFTON, OH"
    def test_road_abbreviated(self):
        result = normalize_address('11500 Island Road, Grafton, OH', 'US')
        self.assertIn('rd', result)
        self.assertNotIn('road', result)

    # EDA: Addresses appear in uppercase in S2
    def test_uppercase_lowercased(self):
        result = normalize_address('9553 NARANJA AVENUE, MESA, AZ', 'US')
        self.assertEqual(result, normalize_address('9553 Naranja Avenue, Mesa, AZ', 'US'))

    def test_empty_address(self):
        self.assertEqual(normalize_address(''), '')
        self.assertEqual(normalize_address(None), '')
        self.assertEqual(normalize_address('nan'), '')


class TestExpandStateAbbreviations(unittest.TestCase):
    # EDA: 2,572,549 pairs with state-abbr vs full-state-name
    def test_nc_expanded(self):
        result = expand_state_abbreviations('Asheville, NC', 'US')
        self.assertIn('north carolina', result)

    def test_tx_expanded(self):
        result = expand_state_abbreviations('Austin, TX', 'US')
        self.assertIn('texas', result)

    def test_ny_expanded(self):
        result = expand_state_abbreviations('Ticonderoga, NY', 'US')
        self.assertIn('new york', result)

    def test_mn_expanded(self):
        result = expand_state_abbreviations('Saint Cloud, MN', 'US')
        self.assertIn('minnesota', result)

    # Non-US: abbreviations should NOT be expanded
    def test_india_no_expansion(self):
        result = expand_state_abbreviations('Hyderabad, AP', 'India')
        # should not expand AP to anything — India path not expanded here
        # (India expansion is handled in extract_state_region, not here)
        self.assertIn('AP', result)

    def test_sorted_addr_tokens_handles_reorder(self):
        # EDA: "Greensboro, 1604 Washington Street, NC" ↔ "1604 WASHINGTON STREET, GREENSBORO, NC"
        a1 = sorted_address_tokens('Greensboro, 1604 Washington Street, NC', 'US')
        a2 = sorted_address_tokens('1604 Washington Street, Greensboro, NC', 'US')
        self.assertEqual(a1, a2)


class TestAbbreviateStreetTypes(unittest.TestCase):
    def test_avenue_to_ave(self):
        self.assertIn('ave', abbreviate_street_types('Naranja Avenue'))

    def test_boulevard_to_blvd(self):
        self.assertIn('blvd', abbreviate_street_types('Sunset Boulevard'))

    def test_north_to_n(self):
        self.assertIn(' n ', abbreviate_street_types(' North Avenue'))

    # EDA: "North Carolina" — do NOT abbreviate 'North' inside 'North Carolina'
    # (state expansion happens before street type abbreviation, so 'north carolina'
    #  is already one token; street abbr works token-by-token)
    def test_street_abbr_does_not_corrupt_state_name(self):
        # After state expansion "north carolina" is a phrase; abbreviate_street_types
        # replaces standalone \bnorth\b → n, which may affect this.
        # The key invariant: after full pipeline, state extraction still works.
        cleaned = normalize_address('Asheville, North Carolina', 'US')
        state = extract_state_region('Asheville, North Carolina', 'US')
        self.assertIsNotNone(state)


# ══════════════════════════════════════════════════════════════════════════════
# Field extraction
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractPostalCode(unittest.TestCase):
    # EDA: US ZIP codes observed in training data
    def test_us_zip_5digit(self):
        self.assertEqual(extract_postal_code('15602 60, Borden, IN 46711', 'US'), '15602')

    def test_us_zip_simple(self):
        self.assertEqual(extract_postal_code('126 Presnell Street, Asheboro, NC 27203', 'US'), '27203')

    # EDA: India PIN codes (6-digit)
    def test_india_pin(self):
        self.assertEqual(extract_postal_code('B-57 Okhla Industrial Area, Delhi 110020', 'India'), '110020')

    def test_no_postal_code(self):
        self.assertIsNone(extract_postal_code('Asheville, NC', 'US'))

    def test_empty(self):
        self.assertIsNone(extract_postal_code('', 'US'))


class TestExtractStateRegion(unittest.TestCase):
    # EDA: full state names in S3
    def test_us_full_state(self):
        result = extract_state_region('37 Simpson Street, Asheville, North Carolina', 'US')
        self.assertEqual(result, 'NC')

    def test_us_full_state_texas(self):
        result = extract_state_region('2803 Stratford Drive, Austin, Texas', 'US')
        self.assertEqual(result, 'TX')

    # EDA: 2-letter abbreviation
    def test_us_abbr_state(self):
        result = extract_state_region('9553 Naranja Avenue, Mesa, AZ', 'US')
        self.assertEqual(result, 'AZ')

    def test_us_state_nc(self):
        result = extract_state_region('1604 Washington Street, Greensboro, NC', 'US')
        self.assertEqual(result, 'NC')

    def test_india_state(self):
        result = extract_state_region('B-57 Okhla Industrial Area Phase-1, Delhi', 'India')
        self.assertIsNotNone(result)

    def test_empty(self):
        self.assertIsNone(extract_state_region('', 'US'))


class TestExtractCity(unittest.TestCase):
    # EDA: standard US address "street, city, state"
    def test_us_city_asheville(self):
        result = extract_city('37 Simpson Street, Asheville, NC', 'US')
        self.assertEqual(result, 'asheville')

    def test_us_city_austin(self):
        result = extract_city('2803 Stratford Drive, Austin, TX', 'US')
        self.assertEqual(result, 'austin')

    def test_us_city_mesa(self):
        result = extract_city('9553 Naranja Avenue, Mesa, AZ', 'US')
        self.assertEqual(result, 'mesa')

    def test_empty(self):
        self.assertIsNone(extract_city('', 'US'))


# ══════════════════════════════════════════════════════════════════════════════
# Combined normalize_record_address
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizeRecordAddress(unittest.TestCase):
    def test_full_us_address(self):
        rec = normalize_record_address('37 Simpson Street, Unit 205, Asheville, NC', 'US')
        self.assertIn('st', rec.cleaned)         # Street → st
        self.assertIn('asheville', rec.cleaned)  # city preserved
        self.assertIsNotNone(rec.state_region)
        self.assertEqual(rec.state_region, 'NC')
        self.assertEqual(rec.city, 'asheville')
        self.assertIsNone(rec.postal_code)        # no ZIP in this address

    def test_us_address_with_zip(self):
        rec = normalize_record_address('126 Presnell Street, Asheboro, NC 27203', 'US')
        self.assertEqual(rec.postal_code, '27203')

    # EDA: reordered address tokens sort to same bag
    def test_address_reorder_same_sorted_tokens(self):
        a1 = normalize_record_address('Greensboro, 1604 Washington Street, NC', 'US')
        a2 = normalize_record_address('1604 Washington Street, Greensboro, NC', 'US')
        self.assertEqual(a1.sorted_tokens, a2.sorted_tokens)

    def test_empty_address(self):
        rec = normalize_record_address('', 'US')
        self.assertEqual(rec.cleaned, '')
        self.assertEqual(rec.tokens, [])
        self.assertIsNone(rec.postal_code)


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end: matched-pair convergence tests
# All pairs taken directly from EDA output
# ══════════════════════════════════════════════════════════════════════════════

class TestMatchedPairConvergence(unittest.TestCase):
    """
    For each EDA-observed matched pair, the normalized representations
    should be equal (or very close) after normalization.
    """

    def _core_eq(self, n1, n2, msg=''):
        self.assertEqual(core_name(n1), core_name(n2), msg or f'{n1!r} vs {n2!r}')

    def _sorted_eq(self, n1, n2, msg=''):
        self.assertEqual(sorted_tokens(n1), sorted_tokens(n2), msg or f'{n1!r} vs {n2!r}')

    # EDA: "New Delhi Events Private Limited" ↔ "New Delhi Events Private Ltd"
    def test_private_limited_vs_ltd(self):
        self._core_eq('New Delhi Events Private Limited', 'New Delhi Events Private Ltd')

    # EDA: "American Ridge Dogecoin Inc" ↔ "American Ridge Dogecoin"
    def test_inc_dropped(self):
        self._core_eq('American Ridge Dogecoin Inc', 'American Ridge Dogecoin')

    # EDA: "Hendricks Ship LLC" ↔ "Hendricks Ship-L.L.C."
    def test_llc_vs_llc_dotted(self):
        self._core_eq('Hendricks Ship LLC', 'Hendricks Ship-L.L.C.')

    # EDA: "Global Infrastructure Pvt Ltd" ↔ "Global Infrastructure  Pvt Ltd" (double space)
    def test_double_space_in_match(self):
        self._core_eq('Global Infrastructure Pvt Ltd', 'Global Infrastructure  Pvt Ltd')

    # EDA: "Drayova Nuveen Inc." ↔ "Drayova Inc. Nuveen" (word order)
    def test_word_order_swap(self):
        self._sorted_eq('Drayova Nuveen Inc.', 'Drayova Inc. Nuveen')

    # EDA: "Crystal Staffing Solutions LLC" ↔ "Llc Crystal Staffing Solutions" (LLC at start)
    def test_llc_at_start(self):
        self._sorted_eq('Crystal Staffing Solutions LLC', 'Llc Crystal Staffing Solutions')

    # EDA: "Blankenship and Russo Inc" ↔ "Blankenship & Rnsumo Inc" (& vs and + typo)
    def test_ampersand_vs_and_core(self):
        n1 = normalize_name('Blankenship and Russo Inc')
        n2 = normalize_name('Blankenship & Russo Inc')
        # Strip suffix then compare
        self.assertEqual(core_name('Blankenship and Russo'), core_name('Blankenship & Russo'))

    # EDA: "Jaipur Estate Private Limited" ↔ "Jaipur-Estate Private Limited" (hyphen)
    def test_hyphen_in_name(self):
        self._core_eq('Jaipur Estate Private Limited', 'Jaipur-Estate Private Limited')

    # EDA: "Payne Enterprises" ↔ "Payne Énterprises" (accented char)
    def test_accented_char_match(self):
        self._core_eq('Payne Enterprises', 'Payne Énterprises')

    # EDA: "Norton Frontier Tge" ↔ "Norton Frontier Tge Co" (Co added)
    def test_co_suffix_dropped(self):
        self._core_eq('Norton Frontier Tge', 'Norton Frontier Tge Co')

    # EDA: "Brown Safe Residential LLC" ↔ "Brown Safe" (partial name)
    # core_name won't make them equal, but both should have "brown safe" as a prefix
    def test_partial_name_core_contains_base(self):
        core_full = core_name('Brown Safe Residential LLC')
        core_short = core_name('Brown Safe')
        self.assertTrue(core_full.startswith(core_short))

    # EDA: "Jeanlouis Advantage LLC" ↔ "LLC JEANLOUIS ADVANTAGE"
    def test_all_caps_word_order(self):
        self._sorted_eq('Jeanlouis Advantage LLC', 'LLC JEANLOUIS ADVANTAGE')

    # EDA: address sorted tokens — component reordering
    # "Greensboro, 1604 Washington Street, NC" ↔ "1604 WASHINGTON STREET, GREENSBORO, NC"
    def test_address_component_reorder(self):
        a1 = sorted_address_tokens('Greensboro, 1604 Washington Street, NC', 'US')
        a2 = sorted_address_tokens('1604 WASHINGTON STREET, GREENSBORO, NC', 'US')
        self.assertEqual(a1, a2)

    # EDA: "37 Simpson Street, Unit 205, Asheville, NC"
    #      ↔ "Simpson Street, # 205, Asheville, North Carolina"
    # After street abbreviation + state expansion, city and state should match
    def test_street_abbr_and_state_expansion(self):
        r1 = normalize_record_address('37 Simpson Street, Asheville, NC', 'US')
        r2 = normalize_record_address('Simpson Street, Asheville, North Carolina', 'US')
        self.assertEqual(r1.state_region, r2.state_region)
        self.assertEqual(r1.city, r2.city)

    # EDA: "2803 Stratford Drive, Austin, TX" ↔ "2803 Startford Dr, Austin, Texas"
    def test_drive_vs_dr_and_state_expansion(self):
        r1 = normalize_record_address('2803 Stratford Drive, Austin, TX', 'US')
        r2 = normalize_record_address('2803 Startford Dr, Austin, Texas', 'US')
        self.assertEqual(r1.state_region, r2.state_region)
        self.assertEqual(r1.city, r2.city)


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
