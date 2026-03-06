from .tracker import Tracker, trackerlist
from .environment import create_default_local_file_ITP_test

# Some lightweight runtime paths (e.g. tracking/video_demo.py) only need Tracker.
# In some checkouts, training data helpers under lib/train/data may be missing,
# which breaks eager imports below. Keep them optional.
try:
    from .data import Sequence
    from .datasets import get_dataset
except ModuleNotFoundError as exc:
    missing_detail = str(exc)
    Sequence = None

    def get_dataset(*args, **kwargs):
        raise ModuleNotFoundError(
            "Dataset utilities are unavailable because required modules are missing: {}"
            .format(missing_detail)
        )
