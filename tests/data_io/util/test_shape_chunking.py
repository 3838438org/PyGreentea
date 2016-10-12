import unittest

from data_io.util.shape_chunking import chunkify_shape


class TestShapeChunking(unittest.TestCase):
    def test_works_with_no_provided_integral_block_shape(self):
        chunk_offsets = chunkify_shape((1,), (1,))
        self.assertEqual(chunk_offsets, ((0,),))

        chunk_offsets = chunkify_shape((2,), (1,))
        self.assertEqual(chunk_offsets, ((0,), (1,)))

        chunk_offsets = chunkify_shape((7,), (5,))
        self.assertEqual(chunk_offsets, ((0,), (2,)))

        chunk_offsets = chunkify_shape((3, 4), (2, 2))
        self.assertEqual(chunk_offsets, ((0, 0), (0, 2), (1, 0), (1, 2)))

    def test_works_with_provided_integral_block_shape(self):
        chunk_offsets = chunkify_shape((1,), (1,), (1,))
        self.assertEqual(chunk_offsets, ((0,),))

        chunk_offsets = chunkify_shape((2,), (1,), (1,))
        self.assertEqual(chunk_offsets, ((0,), (1,)))

        chunk_offsets = chunkify_shape((7,), (5,), (1,))
        self.assertEqual(chunk_offsets, ((0,), (2,)))

        chunk_offsets = chunkify_shape((3, 4), (2, 2), (1, 1))
        self.assertEqual(chunk_offsets, ((0, 0), (0, 2), (1, 0), (1, 2)))

        with self.assertRaises(ValueError) as context:
            chunkify_shape((1,), (1,), (2,))
        self.assertTrue("chunk_shape must be an integer multiple of integral_block_shape along every axis" in context.exception)

        with self.assertRaises(ValueError) as context:
            chunkify_shape((7,), (5,), (2,))
        self.assertTrue("chunk_shape must be an integer multiple of integral_block_shape along every axis" in context.exception)

        chunk_offsets = chunkify_shape((7, 13), (4, 6), (2, 3))
        self.assertEqual(chunk_offsets, ((0, 0), (0, 6), (0, 9), (4, 0), (4, 6), (4, 9)))
