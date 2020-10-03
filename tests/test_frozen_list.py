import unittest

from frozenlist.frozen_list import FrozenList


class MyTestCase(unittest.TestCase):
    def test_creation(self):
        l = [1, 2, 3]
        f = FrozenList(l)
        self.assertEqual(f, l)

    def test_indexing(self):
        l = [1, 2, 3]
        f = FrozenList(l)
        self.assertEqual(f[0], 1)

    def test_slicing(self):
        l = [1, 2, 3]
        f = FrozenList(l)
        self.assertEqual(f[1:2], [2])

    def test_modify_base(self):
        l = [1, 2, 3]
        f = FrozenList(l)
        self.assertEqual(f[0], 1)
        l[0] = 5
        self.assertEqual(f[0], 5)

    def test_modify_base_slice(self):
        l = [1, 2, 3]
        f = FrozenList(l)
        self.assertEqual(f[1:2], [2])
        l[1] = 5
        self.assertEqual(f[1:2], [5])

    def test_cant_mutate_slice(self):
        l = [1, 2, 3]
        f = FrozenList(l)
        with self.assertRaises(TypeError) as context:
            f[1:2] = [5]

    def test_cant_append(self):
        l = [1, 2, 3]
        f = FrozenList(l)
        with self.assertRaises(AttributeError) as context:
            f.append(6)

    def test_cant_extend(self):
        l = [1, 2, 3]
        f = FrozenList(l)
        with self.assertRaises(AttributeError) as context:
            f.extend([6, 7])


if __name__ == '__main__':
    unittest.main()
