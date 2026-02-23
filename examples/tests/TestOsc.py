from mmm_python import *

# instantiate and load the graph
m_s = []


for i in range(7):
    mmm_audio = MMMAudio(128, graph_name="TestOsc", package_name="examples.tests")
    mmm_audio.start_audio() 
    m_s.append(mmm_audio)

m_s[0].stop_audio()  