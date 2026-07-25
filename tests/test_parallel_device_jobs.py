import unittest

from audiotoolkits.evaluation.run import (
    _build_fixed_device_jobs,
    _flatten_chunks,
    _flatten_role_chunks,
)


class FixedDeviceJobsTest(unittest.TestCase):
    def test_assigns_all_chunks_to_one_job_per_device(self):
        chunks = [[index] for index in range(10)]
        jobs = _build_fixed_device_jobs(chunks, [f"cuda:{index}" for index in range(8)])

        self.assertEqual(len(jobs), 8)
        self.assertEqual([job["device"] for job in jobs], [f"cuda:{i}" for i in range(8)])
        self.assertEqual(jobs[0]["chunks"], [[0], [8]])
        self.assertEqual(jobs[1]["chunks"], [[1], [9]])
        self.assertEqual(jobs[7]["chunks"], [[7]])
        self.assertEqual(
            sorted(value for job in jobs for chunk in job["chunks"] for value in chunk),
            list(range(10)),
        )

    def test_does_not_create_idle_workers(self):
        jobs = _build_fixed_device_jobs([["a"], ["b"]], ["cuda:0", "cuda:1", "cuda:2"])

        self.assertEqual(jobs, [
            {"device": "cuda:0", "chunks": [["a"]]},
            {"device": "cuda:1", "chunks": [["b"]]},
        ])

    def test_flatten_helpers_preserve_order(self):
        self.assertEqual(_flatten_chunks([[1, 2], [3]]), [1, 2, 3])
        self.assertEqual(
            _flatten_role_chunks([("gen", [1, 2]), ("src", [3])]),
            [("gen", 1), ("gen", 2), ("src", 3)],
        )


if __name__ == "__main__":
    unittest.main()
