import unittest


def And(x1, x2):
    w1 = 1
    w2 = 1
    b = -1.5
    output = w1*x1 + w2*x2 + b
    if output>0:
        return 1
    else:
        return 0


def Or(x1, x2):
    w1 = 1
    w2 = 1
    b = -0.5
    output = w1 * x1 + w2 * x2 + b
    if output > 0:
        return 1
    else:
        return 0

def Nand(x1, x2):
    w1 = -1
    w2 = -1
    b = 1.5
    output = w1 * x1 + w2 * x2 + b
    if output > 0:
        return 1
    else:
        return 0

class TestStringMethods(unittest.TestCase):

    def test_Nand(self):
        self.assertEqual(Nand(0, 1), 1)
        self.assertEqual(Nand(1, 0), 1)
        self.assertEqual(Nand(0, 0), 1)
        self.assertEqual(Nand(1, 1), 0)

    def test_And(self):
        self.assertEqual(And(1, 1), 1)
        self.assertEqual(And(1, 0), 0)
        self.assertEqual(And(0, 1), 0)
        self.assertEqual(And(0, 0), 0)

    def test_Or(self):
        self.assertEqual(Or(1, 1), 1)
        self.assertEqual(Or(1, 0), 1)
        self.assertEqual(Or(0, 1), 1)
        self.assertEqual(Or(0, 0), 0)

tester = TestStringMethods()
tester.test_And()
tester.test_Nand()
tester.test_Or()
