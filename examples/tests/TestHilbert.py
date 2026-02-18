from mmm_python import *

mmm_audio = MMMAudio(128, graph_name="TestHilbert", package_name="examples.tests")
mmm_audio.start_audio()

mmm_audio.send_float("freq", 150)

mmm_audio.stop_audio()
mmm_audio.plot(2048)
