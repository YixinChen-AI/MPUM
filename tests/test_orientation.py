import math
import unittest

import numpy as np
import SimpleITK as sitk

from datautils import adjust_image_direction, resample_image


RAS = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)


class ImageOrientationTest(unittest.TestCase):
    def test_axis_permutation_and_flip_preserve_physical_labels(self):
        data = np.zeros((12, 9, 7), dtype=np.uint8)
        data[2:5, 1:3, 1:3] = 11
        data[7:10, 5:7, 4:6] = 22
        expected = sitk.GetImageFromArray(data)
        expected.SetDirection(RAS)
        expected.SetSpacing((1.4, 2.0, 3.2))
        expected.SetOrigin((20.0, 60.0, -34.0))

        for orientation in ("RAS", "LPS", "PLS"):
            with self.subTest(orientation=orientation):
                source = sitk.DICOMOrient(expected, orientation)
                adjusted = adjust_image_direction(source, RAS, verbose=False)
                self.assertEqual(adjusted.GetDirection(), expected.GetDirection())
                np.testing.assert_allclose(adjusted.GetOrigin(), expected.GetOrigin(), rtol=0, atol=1e-8)
                self.assertEqual(adjusted.GetSpacing(), expected.GetSpacing())
                np.testing.assert_array_equal(sitk.GetArrayViewFromImage(adjusted), data)
                restored = resample_image(adjusted, source)
                np.testing.assert_array_equal(sitk.GetArrayViewFromImage(restored), sitk.GetArrayViewFromImage(source))

    def test_equivalent_input_encodings_share_the_inference_grid(self):
        data = np.arange(12 * 9 * 7, dtype=np.float32).reshape(12, 9, 7)
        ras = sitk.GetImageFromArray(data)
        ras.SetDirection(RAS)
        ras.SetSpacing((1.4, 2.0, 3.2))
        ras.SetOrigin((20.0, 60.0, -34.0))
        spacing = (2.0, 2.0, 2.0)
        reference = adjust_image_direction(ras, RAS, verbose=False, output_spacing=spacing)
        for encoding in ("LPS", "PLS"):
            with self.subTest(encoding=encoding):
                image = sitk.DICOMOrient(ras, encoding)
                adjusted = adjust_image_direction(image, RAS, verbose=False, output_spacing=spacing)
                self.assertEqual(adjusted.GetSize(), reference.GetSize())
                self.assertEqual(adjusted.GetDirection(), reference.GetDirection())
                self.assertEqual(adjusted.GetSpacing(), spacing)
                np.testing.assert_allclose(adjusted.GetOrigin(), reference.GetOrigin(), rtol=0, atol=1e-8)
                np.testing.assert_allclose(sitk.GetArrayViewFromImage(adjusted),
                                           sitk.GetArrayViewFromImage(reference), rtol=0, atol=1e-6)

    def test_oblique_resampling_preserves_world_coordinates_and_coverage(self):
        source = sitk.Image([11, 9, 7], sitk.sitkFloat32)
        angle = math.radians(24)
        c, s = math.cos(angle), math.sin(angle)
        source.SetDirection((c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0))
        source.SetSpacing((1.25, 2.0, 2.5))
        source.SetOrigin((16.0, 31.0, 5.0))

        def intensity(point):
            x, y, z = point
            return x + 2 * y + 3 * z

        for index in np.ndindex(source.GetSize()):
            source[index] = intensity(source.TransformIndexToPhysicalPoint(index))

        model_spacing = (2.0, 2.0, 2.0)
        adjusted = adjust_image_direction(
            source, RAS, verbose=False, output_spacing=model_spacing, default_value=-1000,
        )
        self.assertEqual(adjusted.GetDirection(), RAS)
        self.assertEqual(adjusted.GetSpacing(), model_spacing)

        for corner in np.ndindex(2, 2, 2):
            original_index = tuple(size - 0.5 if i else -0.5 for i, size in zip(corner, source.GetSize()))
            point = source.TransformContinuousIndexToPhysicalPoint(original_index)
            output_index = np.array(adjusted.TransformPhysicalPointToContinuousIndex(point))
            self.assertTrue(np.all(output_index >= -0.5 - 1e-8))
            self.assertTrue(np.all(output_index <= np.asarray(adjusted.GetSize()) - 0.5 + 1e-8))

        center = tuple(size // 2 for size in adjusted.GetSize())
        self.assertAlmostEqual(adjusted[center], intensity(adjusted.TransformIndexToPhysicalPoint(center)), places=4)
        self.assertIn(-1000.0, sitk.GetArrayViewFromImage(adjusted))


if __name__ == "__main__":
    unittest.main()
