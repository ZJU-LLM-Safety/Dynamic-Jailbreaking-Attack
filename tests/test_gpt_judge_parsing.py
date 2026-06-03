import unittest

from src.attacker_v3 import extract_tagged_int_score


class TestTaggedJudgeParsing(unittest.TestCase):
    def test_extracts_individual_quality_scores(self):
        text = "#specificity: 4\n#relevance: 5\n#coherence: 3\n#nonrefusal: 2\n"

        self.assertEqual(extract_tagged_int_score("#specificity:", text), 4)
        self.assertEqual(extract_tagged_int_score("#relevance:", text), 5)
        self.assertEqual(extract_tagged_int_score("#coherence:", text), 3)
        self.assertEqual(extract_tagged_int_score("#nonrefusal:", text), 2)


if __name__ == "__main__":
    unittest.main()
