from mmm_audio import *

comptime num_bands: Int = 10

struct MelBandsExample(Movable, Copyable):
    var world: LegacyUnsafePointer[MMMWorld]
    var buffer: Buffer
    var playBuf: Play
    var analyzer: FFTProcess[MelBands[num_bands,fft_size=1024],1024,512,WindowType.hann]
    var m: Messenger
    var mul: Float64

    fn __init__(out self, world: LegacyUnsafePointer[MMMWorld]):
        self.world = world
        self.buffer = Buffer.load("resources/Shiverer.wav")
        self.playBuf = Play(self.world)
        p = MelBands[num_bands,fft_size=1024](self.world)
        self.analyzer = FFTProcess[MelBands[num_bands,fft_size=1024],1024,512,WindowType.hann](self.world,p^)
        self.m = Messenger(self.world)
        self.mul = 500.0

    fn next(mut self) -> SIMD[DType.float64, 2]:
        
        self.m.update(self.mul,"multiplier")

        flute = self.playBuf.next(self.buffer)
        
        # do the analysis
        _ = self.analyzer.next(flute)

        # get the results
        if self.world[].top_of_block:
            # print the mel band energies
            string = ""
            for i in range(num_bands):
                val = self.analyzer.buffered_process.process.process.bands[i]
                string += String(val) + " "
                for _ in range(Int(val * self.mul)):
                    string += "*"
                string += "\n"
            
            print(string)
            # print the results
        
        return flute * 0.0
