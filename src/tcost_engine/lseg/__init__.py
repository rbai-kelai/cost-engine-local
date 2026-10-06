"""Pull LSEG Datastream2 panels the same way the AWS research box does.

Canonical artifact (also on S3):

- local on ``kelai-team-robert``: ``/data/robert/lseg/Datastream2/ds2_data.h5``
- S3: ``s3://kelaidata/data/LSEG/Datastream2/ds2_data.h5``

Panels live under ``ds2_data/<FIELD>`` (rows = dates, columns = INFOCODE) with
a ticker vocabulary at ``metadata/TICKERS``. OHLCV is available unadjusted
(``OPEN``…``VOLUME``) and adjusted (``*_ADJUSTED``). Point-in-time universe
flags include ``TOP500``.
"""

from tcost_engine.lseg.paths import DEFAULT_DS2_H5_LOCAL, DEFAULT_DS2_H5_S3, resolve_ds2_h5
from tcost_engine.lseg.pull import pull_cost_prices, pull_ohlcv, pull_top500

__all__ = [
    "DEFAULT_DS2_H5_LOCAL",
    "DEFAULT_DS2_H5_S3",
    "pull_cost_prices",
    "pull_ohlcv",
    "pull_top500",
    "resolve_ds2_h5",
]
