import sys
sys.path.append("..")

from benchmark_driver import BenchmarkMetricComputeMethod
import unittest


class MetricComputeTest(unittest.TestCase):
    def test_metric_compute_average(self):
        metric = [1, 2, 3]
        metric_compute_method = 'average'
        self.assertEqual(BenchmarkMetricComputeMethod.compute(metric_compute_method, metric), 2)

    def test_metric_compute_total(self):
        metric = [1, 2, 3]
        metric_compute_method = 'total'
        self.assertEqual(BenchmarkMetricComputeMethod.compute(metric_compute_method, metric), 6)

    def test_metric_compute_last(self):
        metric = [1, 2, 3]
        metric_compute_method = 'last'
        self.assertEqual(BenchmarkMetricComputeMethod.compute(metric_compute_method, metric), 3)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(MetricComputeTest)
    unittest.TextTestRunner(verbosity=2).run(suite)