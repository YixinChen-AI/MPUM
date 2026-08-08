import unittest
from types import SimpleNamespace

from dynamic_pet.adapter import complete_frame_groups, injection_datetime, injection_datetime_source


class DynamicPetMetadataTest(unittest.TestCase):
    def test_conflicting_full_injection_date_falls_back_to_series_date(self):
        dataset = SimpleNamespace(SeriesDate="20200902")
        radiopharm = SimpleNamespace(
            RadiopharmaceuticalStartDateTime="20190620080100",
            RadiopharmaceuticalStartTime="080100",
        )
        value = injection_datetime(dataset, radiopharm)
        self.assertEqual(value.strftime("%Y%m%d%H%M%S"), "20200902080100")
        self.assertIn("date conflict", injection_datetime_source(dataset, radiopharm))

    def test_incomplete_last_frame_is_excluded(self):
        def item(time, duration=30000):
            return (None, SimpleNamespace(FrameReferenceTime=time, ActualFrameDuration=duration))

        items = [item(0) for _ in range(71)] + [item(30000) for _ in range(71)] + [item(60000) for _ in range(43)]
        complete, discarded = complete_frame_groups(items)
        self.assertEqual(sorted(complete), [0.0, 30000.0])
        self.assertEqual(discarded[0]["slice_count"], 43)
        self.assertEqual(discarded[0]["expected_slice_count"], 71)


if __name__ == "__main__":
    unittest.main()
