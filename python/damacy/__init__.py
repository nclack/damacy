"""High-throughput streaming loader for sharded NGFF zarr stores.

The native extension is loaded eagerly so the in-process log sink is
registered as soon as `damacy` is imported. Log records produced by C
threads are routed to `logging.getLogger("damacy")` via a daemon-thread
drain that holds the GIL only while delivering messages.
"""

from damacy import _native

__all__ = ["_native"]
__version__ = _native.__version__
