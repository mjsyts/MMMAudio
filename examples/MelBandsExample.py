"""
This example demonstrates how to use MelBands
"""

from mmm_python import *
ma = MMMAudio(128, graph_name="MelBandsExample", package_name="examples")
ma.start_audio()

ma.send_float("multiplier",500.0) # 500 is the default in Mojo also

ma.stop_audio()