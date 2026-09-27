"""
test_france_smoke.py
--------------------
Smoke test for unseen-country (France) handling across the full pipeline:
  normalize -> blocking -> features -> model predict

Uses only synthetic French-looking records — no external data, no internet.
Verifies no crashes and that France records are handled consistently with
US/India records, using the same code paths.

Run with:
    python test_france_smoke.py
    or: python -m pytest test_france_smoke.py -v
"""
import os
import sys
import json
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from normalize import (
    normalize_name,
    normalize_address,
    core_name,
    sorted_tokens,
    phonetic_name,
    expand_state_abbreviations,
    extract_postal_code,
    extract_state_region,
    extract_city,
    remove_accents,
    _FR_REGION_MAP,
)
from blocking import BlockingConfig, BlockingEngine, _country_matches
from features import FeatureBuilder, MISSING_VAL, ALL_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic French records (no real data from test set)
# ─────────────────────────────────────────────────────────────────────────────

FRENCH_NAMES = [
    # Accented names
    ("Marina Écoles S.A.S",        "Marina Ecoles SAS"),
    ("Société Générale",            "Societe Generale"),
    ("Boulangerie Chez André",      "Boulangerie Chez Andre"),
    # Legal suffixes present in training
    ("Dupont Frères S.A.",          "Dupont Freres SA"),
    ("Tech Paris SARL",             "Tech Paris"),
    ("Innovatech S.A.S.",           "INNOVATECH SAS"),
    # Word-order variants
    ("Paris Café du Centre",        "Cafe du Centre Paris"),
    # Completely different name (should not match)
    ("Maison Lefebvre",             "Bureau Martin"),
]

FRENCH_ADDRESSES = [
    # Standard French format: number rue, postal city, region
    "12 Rue de la Paix, 75001 Paris, Île-de-France",
    "63 R. DE DIEPPE, LILLE, Hauts-de-France",
    "18 RUE JEN ZAY, Dunkerque, Nord",
    "23 Rue Icmre, La Teste-de-Buch, Gironde",
    # With IDF abbreviation
    "7 Rue du Louvre, 75001 Paris, IDF",
    # No region
    "45 Avenue des Champs-Elysées, 75008 Paris",
    # Minimal
    "Paris",
    # Empty
    "",
]


def _make_df(rows):
    """Create a minimal DataFrame from list of (entity_id, name, addr, country) tuples."""
    return pd.DataFrame(rows, columns=['entity_id', 'business_name',
                                        'business_address', 'country'])


# ─────────────────────────────────────────────────────────────────────────────
# Test classes
# ─────────────────────────────────────────────────────────────────────────────

class TestFranceNormalize(unittest.TestCase):
    """normalize.py: French records must not crash and must produce clean output."""

    def test_accented_names_decoded(self):
        """Accented Latin characters are stripped for matching."""
        self.assertEqual(remove_accents('É'), 'E')
        self.assertEqual(remove_accents('é'), 'e')
        self.assertEqual(remove_accents('ç'), 'c')
        self.assertEqual(remove_accents('à'), 'a')
        self.assertEqual(remove_accents('Île'), 'Ile')

    def test_french_sarl_sas_normalised(self):
        """SARL and SAS are in the legal suffix table."""
        self.assertEqual(core_name('Tech Paris SARL'), 'tech paris')
        self.assertEqual(core_name('Innovatech S.A.S.'), 'innovatech')
        self.assertEqual(core_name('Dupont Frères S.A.'), 'dupont freres')

    def test_no_crash_on_all_french_names(self):
        """normalize_name must not crash on any French-looking input."""
        for raw, _ in FRENCH_NAMES:
            result = normalize_name(raw)
            self.assertIsInstance(result, str)
            self.assertNotIn('nan', result)

    def test_french_address_normalize_no_crash(self):
        """normalize_address must not crash on French addresses."""
        for addr in FRENCH_ADDRESSES:
            result = normalize_address(addr, country='France')
            self.assertIsInstance(result, str)

    def test_us_state_not_expanded_for_france(self):
        """'OR' in a French address must NOT be expanded to 'oregon'."""
        # 'OR' could appear as part of a French word or abbreviation
        result = expand_state_abbreviations('OR blah', country='France')
        self.assertNotIn('oregon', result.lower(),
                         "US state expansion must be skipped for country=France")

    def test_us_state_not_expanded_for_blank_country(self):
        """Blank country must NOT trigger US state expansion (protects French data)."""
        # A French address with a 2-letter token that matches a US state code
        result = expand_state_abbreviations('75001 Paris CA', country='')
        # 'CA' should NOT be expanded to 'california' when country is blank
        self.assertNotIn('california', result.lower(),
                         "US state expansion must be skipped for blank country")

    def test_french_postal_code_extracted(self):
        """5-digit French postal codes are extracted by the generic fallback."""
        result = extract_postal_code('12 Rue de la Paix, 75001 Paris', country='France')
        self.assertEqual(result, '75001')

    def test_french_postal_code_extracted_blank_country(self):
        """5-digit postal code extracted even without country hint."""
        result = extract_postal_code('12 Rue de la Paix, 75001 Paris', country='')
        self.assertIsNotNone(result)
        self.assertEqual(result, '75001')

    def test_fr_region_map_used_in_extract_state(self):
        """_FR_REGION_MAP abbreviations are now wired into extract_state_region."""
        result = extract_state_region('7 Rue du Louvre, Paris, IDF', country='France')
        self.assertEqual(result, _FR_REGION_MAP['IDF'],
                         "IDF should expand to 'ile-de-france'")

    def test_hauts_de_france_as_region(self):
        """Full French region name returned via generic fallback."""
        result = extract_state_region(
            '63 R. DE DIEPPE, LILLE, Hauts-de-France', country='France')
        self.assertIsNotNone(result)
        self.assertIn('france', result.lower())

    def test_extract_state_blank_country_no_us_state(self):
        """Blank country must not return a US state abbr for French address."""
        result = extract_state_region('12 Rue de la Paix, 75001 Paris, OR', country='')
        # With blank country, the US branch no longer fires, so 'OR' won't be
        # returned as a US state abbreviation; fallback returns last comma segment.
        if result is not None:
            # Should not be a known US state abbreviation if it came from a
            # non-US address — it might still return 'OR' as a generic segment,
            # but it must not claim to expand it as "oregon"
            self.assertNotEqual(result, 'OR',
                "Generic fallback returns lowercase; 'OR' as US abbr was uppercase")

    def test_city_extracted_france(self):
        """City extraction returns a result for standard French address."""
        result = extract_city('12 Rue de la Paix, 75001 Paris, Île-de-France',
                              country='France')
        self.assertIsNotNone(result)
        # Should return something reasonable — middle segment or last non-postal
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_accent_insensitive_name_match(self):
        """Accented and unaccented variants share the same core_name."""
        n1 = core_name('Marina Écoles S.A.S')
        n2 = core_name('Marina Ecoles SAS')
        self.assertEqual(n1, n2,
                         f"Accent-stripped names must match: {n1!r} vs {n2!r}")

    def test_phonetic_no_crash_french(self):
        """phonetic_name must not crash on French-accented input."""
        for raw, _ in FRENCH_NAMES:
            result = phonetic_name(raw)
            self.assertIsInstance(result, str)


class TestFranceBlocking(unittest.TestCase):
    """blocking.py: France records flow through correctly without crashes."""

    def setUp(self):
        # Build a tiny synthetic dataset with French, US, and India records
        s1_rows = [
            ('S1-F001', 'Marina Écoles S.A.S', '63 R. DE DIEPPE, LILLE, Hauts-de-France', 'France'),
            ('S1-F002', 'Dupont Frères S.A.', '12 Rue de la Paix, 75001 Paris', 'France'),
            ('S1-F003', 'Société Générale SARL', '45 Avenue des Champs-Elysées, Paris', 'France'),
            ('S1-U001', 'Acme Corporation', '123 Main Street, Springfield, IL', 'US'),
            ('S1-I001', 'Raj Investments LLP', 'B-57 Okhla Industrial Area, Delhi', 'India'),
        ]
        s2_rows = [
            ('S2-F001', 'Marina Ecoles SAS', '63 RUE DE DIEPPE, LILLE, Hauts-de-France', 'France'),
            ('S2-F002', 'Dupont Freres SA', '12 RUE DE LA PAIX, PARIS', 'France'),
            ('S2-U001', 'ACME Corp', '123 MAIN ST, SPRINGFIELD, IL', 'US'),
            ('S2-I001', 'Raj Investments', 'B-57 OKHLA INDUSTRIAL AREA, DELHI', 'India'),
        ]
        s3_rows = [
            ('S3-F001', 'Societe Generale', '45 AVE DES CHAMPS ELYSEES, PARIS', 'France'),
            ('S3-F002', 'Marina Ecoles', 'LILLE, Hauts-de-France', 'France'),
            ('S3-U001', 'Acme Inc', '123 Main St, Springfield, Illinois', 'US'),
        ]
        self.s1 = _make_df(s1_rows)
        self.s2 = _make_df(s2_rows)
        self.s3 = _make_df(s3_rows)
        self.cfg = BlockingConfig()

    def test_blocking_engine_builds_without_crash(self):
        """BlockingEngine.__init__ must not crash on French records."""
        eng = BlockingEngine(self.s1, self.s2, self.s3, self.cfg)
        self.assertIn('S1-F001', eng._s1)
        self.assertIn('S2-F001', eng._s23)
        self.assertIn('S3-F001', eng._s23)

    def test_from_s23_only_france(self):
        """BlockingEngine.from_s23_only must not crash with French S23."""
        eng = BlockingEngine.from_s23_only(self.s2, self.s3, self.cfg, verbose=False)
        self.assertIn('S2-F001', eng._s23)
        self.assertIn('S3-F001', eng._s23)

    def test_candidates_generated_for_france(self):
        """generate_candidates must return results for French S1 entities."""
        eng = BlockingEngine(self.s1, self.s2, self.s3, self.cfg)
        cands = eng.generate_candidates(verbose=False)

        # Every S1 entity must have an entry (even if empty set)
        self.assertIn('S1-F001', cands)
        self.assertIn('S1-F002', cands)
        self.assertIn('S1-F003', cands)

    def test_french_candidates_contain_french_s23(self):
        """French S1 entities should find French S23 candidates."""
        eng = BlockingEngine(self.s1, self.s2, self.s3, self.cfg)
        cands = eng.generate_candidates(verbose=False)

        # S1-F001 'Marina Écoles S.A.S' should match S2-F001 or S3-F002
        french_cands = cands.get('S1-F001', set())
        self.assertTrue(
            'S2-F001' in french_cands or 'S3-F002' in french_cands,
            f"French S1-F001 got no French candidates: {french_cands}"
        )

    def test_french_not_matched_to_us_records(self):
        """French S1 entities should not be paired with US/India S23 records."""
        eng = BlockingEngine(self.s1, self.s2, self.s3, self.cfg)
        cands = eng.generate_candidates(verbose=False)

        for s1_id in ('S1-F001', 'S1-F002', 'S1-F003'):
            for s23_id in cands.get(s1_id, set()):
                country = eng._s23[s23_id]['country']
                self.assertEqual(
                    country.lower(), 'france',
                    f"{s1_id} got cross-country candidate {s23_id} (country={country!r})"
                )

    def test_country_matches_helper_france(self):
        """_country_matches is country-agnostic."""
        self.assertTrue(_country_matches('France', 'France'))
        self.assertTrue(_country_matches('france', 'FRANCE'))
        self.assertFalse(_country_matches('France', 'India'))
        self.assertFalse(_country_matches('France', 'US'))
        self.assertTrue(_country_matches('', 'France'))   # blank = wildcard
        self.assertTrue(_country_matches('France', ''))


class TestFranceFeatures(unittest.TestCase):
    """features.py: feature generation must not crash on French records."""

    def setUp(self):
        s1_rows = [
            ('S1-F001', 'Marina Écoles S.A.S',  '63 R. DE DIEPPE, LILLE, Hauts-de-France', 'France'),
            ('S1-F002', 'Dupont Frères S.A.',    '12 Rue de la Paix, 75001 Paris',           'France'),
            ('S1-F003', 'Boulangerie du Marché', '',                                          'France'),
        ]
        s2_rows = [
            ('S2-F001', 'Marina Ecoles SAS',  '63 RUE DE DIEPPE, LILLE',  'France'),
            ('S2-F002', 'DUPONT FRERES',      '12 RUE DE LA PAIX, PARIS', 'France'),
        ]
        s3_rows = [
            ('S3-F001', 'Societe Generale',    '45 AVE DES CHAMPS ELYSEES, PARIS', 'France'),
        ]
        self.s1 = _make_df(s1_rows)
        self.s2 = _make_df(s2_rows)
        self.s3 = _make_df(s3_rows)
        self.cfg = BlockingConfig()

    def test_feature_builder_no_crash(self):
        """FeatureBuilder must initialise without crashing on French data."""
        fb = FeatureBuilder(self.s1, self.s2, self.s3, verbose=False)
        self.assertIn('S1-F001', fb._s1_cache)
        self.assertIn('S2-F001', fb._s23_cache)

    def test_features_generated_no_nan(self):
        """Feature matrix must contain no NaN values (sentinel -1 is used instead)."""
        eng = BlockingEngine(self.s1, self.s2, self.s3, self.cfg)
        cands = eng.generate_candidates(verbose=False)
        fb = FeatureBuilder(self.s1, self.s2, self.s3, verbose=False)
        X, meta = fb.build_pair_features(cands, verbose=False)

        self.assertEqual(X.shape[1], len(ALL_FEATURES),
                         f"Expected {len(ALL_FEATURES)} features, got {X.shape[1]}")
        self.assertFalse(np.isnan(X).any(),
                         "Feature matrix must not contain NaN")
        # Sentinel value -1.0 is allowed (missing address fields)
        self.assertTrue((X >= -1.0).all(),
                        "No feature value should be below -1.0")
        self.assertTrue((X <= 1.0).all(),
                        "No similarity feature should exceed 1.0")

    def test_country_eq_feature_for_france(self):
        """country_eq feature is 1.0 for France-France pairs, 0.0 for cross-country."""
        from features import _country_eq
        self.assertEqual(_country_eq('France', 'France'), 1.0)
        self.assertEqual(_country_eq('France', 'US'),     0.0)
        self.assertEqual(_country_eq('France', 'India'),  0.0)
        self.assertEqual(_country_eq('France', ''),       0.5)  # unknown = 0.5
        self.assertEqual(_country_eq('',       'France'), 0.5)

    def test_french_address_features_no_crash_empty_addr(self):
        """Records with empty French addresses must not crash feature generation."""
        eng = BlockingEngine(self.s1, self.s2, self.s3, self.cfg)
        cands = eng.generate_candidates(verbose=False)
        fb = FeatureBuilder(self.s1, self.s2, self.s3, verbose=False)
        X, meta = fb.build_pair_features(cands, verbose=False)
        # S1-F003 has empty address; check pairs involving it
        f003_pairs = meta[meta['s1_id'] == 'S1-F003']
        if len(f003_pairs) > 0:
            X_f003 = X[f003_pairs.index]
            self.assertFalse(np.isnan(X_f003).any())
            # s1_addr_missing feature should be 1.0 for S1-F003
            s1_miss_idx = ALL_FEATURES.index('s1_addr_missing')
            self.assertTrue((X_f003[:, s1_miss_idx] == 1.0).all(),
                            "s1_addr_missing must be 1.0 when address is empty")


class TestFranceEndToEnd(unittest.TestCase):
    """End-to-end: load model and score French pairs without crashing."""

    def test_model_predict_on_french_features(self):
        """Trained LightGBM model must score French feature rows without crashing."""
        _MODELS = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', 'models'))
        model_path = os.path.join(_MODELS, 'lgbm_model.txt')
        meta_path  = os.path.join(_MODELS, 'feature_meta.json')

        if not os.path.exists(model_path):
            self.skipTest("lgbm_model.txt not found — run train.py first")

        import lightgbm as lgb
        booster = lgb.Booster(model_file=model_path)
        with open(meta_path) as f:
            fm = json.load(f)

        sentinel = fm['missing_sentinel']

        s1_rows = [
            ('S1-F001', 'Marina Écoles S.A.S',  '63 R. DE DIEPPE, LILLE, Hauts-de-France', 'France'),
            ('S1-F002', 'Dupont Frères S.A.',    '12 Rue de la Paix, 75001 Paris',           'France'),
            ('S1-F003', 'Société Générale SARL', '45 Avenue des Champs-Elysées, Paris',      'France'),
        ]
        s2_rows = [
            ('S2-F001', 'Marina Ecoles SAS',     '63 RUE DE DIEPPE, LILLE', 'France'),
            ('S2-F002', 'DUPONT FRERES',          '12 RUE DE LA PAIX, PARIS', 'France'),
        ]
        s3_rows = [
            ('S3-F001', 'Societe Generale',       '45 AVE DES CHAMPS ELYSEES, PARIS', 'France'),
        ]
        s1 = _make_df(s1_rows)
        s2 = _make_df(s2_rows)
        s3 = _make_df(s3_rows)

        cfg  = BlockingConfig()
        eng  = BlockingEngine(s1, s2, s3, cfg)
        cands = eng.generate_candidates(verbose=False)
        n_cands = sum(len(v) for v in cands.values())

        if n_cands == 0:
            self.skipTest("No candidates generated for French micro-set")

        fb = FeatureBuilder(s1, s2, s3, verbose=False)
        X, meta = fb.build_pair_features(cands, verbose=False)

        # Convert sentinel to NaN as the model expects
        Xf = X.astype(np.float32).copy()
        Xf[Xf == sentinel] = np.nan

        probs = booster.predict(Xf, num_iteration=fm['best_round'])

        # Sanity checks
        self.assertEqual(len(probs), n_cands,
                         "One probability per candidate pair")
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all(),
                        "All probabilities must be in [0, 1]")
        self.assertFalse(np.isnan(probs).any(),
                         "Probabilities must not be NaN")

        print(f"\n    French micro-set: {n_cands} pairs scored")
        print(f"    Prob range: [{probs.min():.4f}, {probs.max():.4f}]")
        for i, row in meta.iterrows():
            print(f"    {row['s1_id']} -> {row['s23_id']}: "
                  f"p={probs[i]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
