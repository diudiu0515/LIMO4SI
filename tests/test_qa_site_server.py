import unittest

from limo4si.qa_site_server import parse_byte_range


class QaSiteServerTests(unittest.TestCase):
    def test_parses_video_byte_ranges(self):
        self.assertEqual(parse_byte_range("bytes=0-1023", 5000), (0, 1023))
        self.assertEqual(parse_byte_range("bytes=1000-", 5000), (1000, 4999))
        self.assertEqual(parse_byte_range("bytes=-500", 5000), (4500, 4999))
        self.assertEqual(parse_byte_range(None, 5000), None)

    def test_rejects_unsupported_or_unsatisfiable_ranges(self):
        for value in ("bytes=5000-6000", "bytes=10-9", "bytes=0-1,3-4", "items=0-1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_byte_range(value, 5000)


if __name__ == "__main__":
    unittest.main()
