class MetricBase:
    name = "base"
    requires_gt_audio = False
    requires_ref_text = False
    supports_src = True

    def __init__(self, cfg):
        self.cfg = cfg

    def prepare(self, context):
        return None

    def compute(self, item, context, role="gen"):
        raise NotImplementedError

    def summary(self):
        return {}
