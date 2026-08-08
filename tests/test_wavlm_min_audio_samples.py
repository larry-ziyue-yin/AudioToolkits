import unittest
from types import SimpleNamespace

from audiotoolkits.evaluation.metrics.wavlm_sim import WavLMSimMetric


class WavLMMinimumAudioSamplesTest(unittest.TestCase):
    def setUp(self):
        config = SimpleNamespace(
            conv_kernel=[10, 3, 3, 3, 3, 2, 2],
            conv_stride=[5, 2, 2, 2, 2, 2, 2],
            tdnn_kernel=[5, 3, 3, 1, 1],
            tdnn_dilation=[1, 2, 3, 1, 1],
        )
        self.metric = WavLMSimMetric({})
        self.metric.model = SimpleNamespace(config=config)

    def test_accumulates_sequential_tdnn_receptive_fields(self):
        self.assertEqual(self.metric._infer_min_frame_length(), 16)

    def test_converts_minimum_frames_back_to_waveform_samples(self):
        self.assertEqual(self.metric._infer_min_audio_samples(), 5200)


if __name__ == "__main__":
    unittest.main()
