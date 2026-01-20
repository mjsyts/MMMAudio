from .MMMWorld_Module import MMMWorld
from .FFTProcess_Module import *
from .FFTs import RealFFT
from math import ceil, floor, log2
from .functions import cpsmidi, ampdb
from math import sqrt

@doc_private
fn parabolic_refine(prev: Float64, cur: Float64, next: Float64) -> (Float64, Float64):
    denom = prev - 2.0 * cur + next
    if abs(denom) < 1e-12:
        return (0.0, cur)
    p = 0.5 * (prev - next) / denom
    refined_val = cur - 0.25 * (prev - next) * p
    return (p, refined_val)

struct YIN[window_size: Int, min_freq: Float64 = 20, max_freq: Float64 = 20000](BufferedProcessable):
    """Monophonic Frequency ('F0') Detection using the YIN algorithm (FFT-based, O(N log N) version).

    Parameters:
        window_size: The size of the analysis window in samples.
        min_freq: The minimum frequency to consider for pitch detection.
        max_freq: The maximum frequency to consider for pitch detection.
    """
    var world: UnsafePointer[MMMWorld]
    var pitch: Float64
    var confidence: Float64
    var sample_rate: Float64
    var fft: RealFFT[window_size * 2]
    var fft_input: List[Float64]
    var fft_power_mags: List[Float64]
    var fft_zero_phases: List[Float64]
    var acf_real: List[Float64]
    var yin_buffer: List[Float64]
    var yin_values: List[Float64]

    fn __init__(out self, world: UnsafePointer[MMMWorld]):
        """Initialize the YIN pitch detector.

        Args:
            world: A pointer to the MMMWorld.

        Returns:
            An initialized YIN struct.
        """
        self.world = world
        self.pitch = 0.0
        self.confidence = 0.0
        self.sample_rate = self.world[].sample_rate
        self.fft = RealFFT[window_size * 2]()
        self.fft_input = List[Float64](length=window_size * 2, fill=0.0)
        self.fft_power_mags = List[Float64](length=window_size + 1, fill=0.0)
        self.fft_zero_phases = List[Float64](length=window_size + 1, fill=0.0)
        self.acf_real = List[Float64](length=window_size * 2, fill=0.0)
        self.yin_buffer = List[Float64](length=window_size, fill=0.0)
        self.yin_values = List[Float64](length=window_size, fill=0.0)
    
    fn next_window(mut self, mut frame: List[Float64]):
        """Compute the YIN pitch estimate for the given frame of audio samples.

        Args:
            frame: The input audio frame of size `window_size`. This List gets passed from [BufferedProcess](BufferedProcess.md).
        """

        # 1. Prepare input for FFT (Zero padding)
        for i in range(len(frame)):
            self.fft_input[i] = frame[i]
        for i in range(len(frame), len(self.fft_input)):
            self.fft_input[i] = 0.0
        
        # 2. FFT
        self.fft.fft(self.fft_input)
        
        # 3. Power Spectrum (Mags^2)
        # We use a separate buffer for power mags so we preserve fft_mags for external use
        for i in range(len(self.fft.mags)):
            self.fft_power_mags[i] = self.fft.mags[i] * self.fft.mags[i]
            
        # 4. IFFT -> Autocorrelation
        # Use zero phases for autocorrelation
        self.fft.ifft(self.fft_power_mags, self.fft_zero_phases, self.acf_real)
        
        # 5. Compute Difference Function
        var total_energy = self.acf_real[0]
        
        var running_sum = 0.0
        for i in range(len(frame)):
            running_sum += frame[i] * frame[i]
            self.yin_buffer[i] = running_sum
            
        self.yin_values[0] = 1.0 
        
        for tau in range(1, len(frame)):
             var term1 = self.yin_buffer[len(frame) - 1 - tau]
             var term2 = total_energy
             if tau > 0:
                 term2 -= self.yin_buffer[tau - 1]
             var term3 = 2.0 * self.acf_real[tau]
             
             self.yin_values[tau] = term1 + term2 - term3

        # cumulative mean normalized difference function
        var tmp_sum: Float64 = 0.0
        for i in range(1, len(frame)):
            raw_val = self.yin_values[i]
            tmp_sum += raw_val
            if tmp_sum != 0.0:
                self.yin_values[i] = raw_val * (Float64(i) / tmp_sum)
            else:
                self.yin_values[i] = 1.0

        var local_pitch = 0.0
        var local_conf = 0.0
        if tmp_sum > 0.0:
            var high_freq = max_freq if max_freq > 0.0 else 1.0
            var low_freq = min_freq if min_freq > 0.0 else 1.0
            
            var min_bin = Int((self.sample_rate / high_freq) + 0.5)
            var max_bin = Int((self.sample_rate / low_freq) + 0.5)

            # Clamp min_bin
            if min_bin < 1:
                min_bin = 1

            # Clamp max_bin
            var safe_limit = len(frame) // 2
            if max_bin > safe_limit:
                max_bin = safe_limit

            if max_bin > min_bin:
                var best_tau = -1
                var best_val = 1.0
                var threshold: Float64 = 0.1
                var tau = min_bin
                while tau < max_bin:
                    var val = self.yin_values[tau]
                    if val < threshold:
                        while tau + 1 < max_bin and self.yin_values[tau + 1] < val:
                            tau += 1
                            val = self.yin_values[tau]
                        best_tau = tau
                        best_val = val
                        break
                    if val < best_val:
                        best_tau = tau
                        best_val = val
                    tau += 1

                if best_tau > 0:
                    var refined_idx = Float64(best_tau)
                    if best_tau > 0 and best_tau < len(frame) - 1:
                        var prev = self.yin_values[best_tau - 1]
                        var cur = self.yin_values[best_tau]
                        var nxt = self.yin_values[best_tau + 1]
                        var (offset, refined_val) = parabolic_refine(prev, cur, nxt)
                        refined_idx += offset
                        best_val = refined_val

                    if refined_idx > 0.0:
                        local_pitch = self.sample_rate / refined_idx
                        local_conf = max(1.0 - best_val, 0.0)
                        local_conf = min(local_conf, 1.0)

        self.pitch = local_pitch
        self.confidence = local_conf

struct SpectralCentroid[min_freq: Float64 = 20, max_freq: Float64 = 20000, power_mag: Bool = False](FFTProcessable):
    """Spectral Centroid analysis.

    Based on the [Peeters (2003)](http://recherche.ircam.fr/anasyn/peeters/ARTICLES/Peeters_2003_cuidadoaudiofeatures.pdf)

    Parameters:
        min_freq: The minimum frequency (in Hz) to consider when computing the centroid.
        max_freq: The maximum frequency (in Hz) to consider when computing the centroid.
        power_mag: If True, use power magnitudes (squared) for the centroid calculation.

    """

    var world: UnsafePointer[MMMWorld]
    var centroid: Float64

    fn __init__(out self, world: UnsafePointer[MMMWorld]):
        self.world = world
        self.centroid = 0.0

    fn next_frame(mut self, mut mags: List[Float64], mut phases: List[Float64]) -> None:
        """Compute the spectral centroid for a given FFT analysis.

        This function is to be used by FFTProcess if SpectralCentroid is passed as the "process".

        Args:
            mags: The input magnitudes as a List of Float64.
            phases: The input phases as a List of Float64.
        """
        self.centroid = self.from_mags(mags, self.world[].sample_rate)

    @staticmethod
    fn from_mags(mags: List[Float64], sample_rate: Float64) -> Float64:
        """Compute the spectral centroid for the given magnitudes of an FFT frame.

        This static method is useful when there is an FFT already computed, perhaps as 
        part of a custom struct that implements the [FFTProcessable](FFTProcess.md/#trait-fftprocessable) trait.

        Args:
            mags: The input magnitudes as a List of Float64.
            sample_rate: The sample rate of the audio signal.

        Returns:
            Float64. The spectral centroid value.
        """
        fft_size: Int = (len(mags) - 1) * 2
        binHz: Float64 = sample_rate / fft_size

        min_bin = Int(ceil(min_freq / binHz))
        max_bin = Int(floor(max_freq / binHz))
        
        min_bin = max(min_bin, 0)
        max_bin = min(max_bin, fft_size // 2)
        max_bin = max(max_bin, min_bin)

        centroid: Float64 = 0.0
        ampsum: Float64 = 0.0

        for i in range(min_bin, max_bin):
            f: Float64 = i * binHz

            m: Float64 = mags[i]

            @parameter
            if power_mag:
                m = m * m

            ampsum += m
            centroid += m * f

        if ampsum > 0.0:
            centroid /= ampsum
        else:
            centroid = 0.0

        return centroid

struct RMS(BufferedProcessable):
    """Root Mean Square (RMS) amplitude analysis.
    """
    var rms: Float64

    fn __init__(out self):
        """Initialize the RMS analyzer."""
        self.rms = 0.0

    fn next_window(mut self, mut input: List[Float64]):
        """Compute the RMS for the given window of audio samples.

        This function is to be used with a [BufferedProcess](BufferedProcess.md/#struct-bufferedprocess).

        Args:
            input: The input audio frame of samples. This List gets passed from [BufferedProcess](BufferedProcess.md/#struct-bufferedprocess).
        
        The computed RMS value is stored in self.rms.
        """
        self.rms = self.from_window(input)

    @staticmethod
    fn from_window(mut frame: List[Float64]) -> Float64:
        """Compute the RMS for the given window of audio samples.

        This static method is useful when there is an audio frame already available, perhaps
        as part of a custom struct that implements the [BufferedProcessable](BufferedProcess.md/#trait-bufferedprocessable) trait.

        Args:
            frame: The input audio frame of samples.
        
        Returns:
            Float64. The computed RMS value.
        """
        sum_sq: Float64 = 0.0
        for v in frame:
            sum_sq += v * v
        return sqrt(sum_sq / Float64(len(frame)))

@doc_private
fn fft_frequencies(sr: Float64, n_fft: Int) -> List[Float64]:
    """Compute the FFT bin center frequencies.

    Args:
        sr: The sample rate of the audio signal.
        n_fft: The size of the FFT.

    Returns:
        A List of Float64 representing the center frequencies of each FFT bin.
    """
    # [TODO] test against: np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    num_bins = (n_fft // 2) + 1
    binHz = sr / Float64(n_fft)
    freqs = List[Float64](length=num_bins, fill=0.0)
    for i in range(num_bins):
        freqs[i] = Float64(i) * binHz
    return freqs^

fn mel_frequencies(
    n_mels: Int = 128, fmin: Float64 = 0.0, fmax: Float64 = 11025.0, htk: Bool = False
) -> List[Float64]:
    """Compute an array of acoustic frequencies tuned to the mel scale.

    The mel scale is a quasi-logarithmic function of acoustic frequency
    designed such that perceptually similar pitch intervals (e.g. octaves)
    appear equal in width over the full hearing range.

    Because the definition of the mel scale is conditioned by a finite number
    of subjective psychoacoustical experiments, several implementations coexist
    in the audio signal processing literature [#]_. By default, librosa replicates
    the behavior of the well-established MATLAB Auditory Toolbox of Slaney [#]_.
    According to this default implementation,  the conversion from Hertz to mel is
    linear below 1 kHz and logarithmic above 1 kHz. Another available implementation
    replicates the Hidden Markov Toolkit [#]_ (HTK) according to the following formula::

        mel = 2595.0 * np.log10(1.0 + f / 700.0).

    The choice of implementation is determined by the ``htk`` keyword argument: setting
    ``htk=False`` leads to the Auditory toolbox implementation, whereas setting it ``htk=True``
    leads to the HTK implementation.

    .. [#] Umesh, S., Cohen, L., & Nelson, D. Fitting the mel scale.
        In Proc. International Conference on Acoustics, Speech, and Signal Processing
        (ICASSP), vol. 1, pp. 217-220, 1998.

    .. [#] Slaney, M. Auditory Toolbox: A MATLAB Toolbox for Auditory
        Modeling Work. Technical Report, version 2, Interval Research Corporation, 1998.

    .. [#] Young, S., Evermann, G., Gales, M., Hain, T., Kershaw, D., Liu, X.,
        Moore, G., Odell, J., Ollason, D., Povey, D., Valtchev, V., & Woodland, P.
        The HTK book, version 3.4. Cambridge University, March 2009.

    Parameters
    ----------
    n_mels : Int > 0 [scalar]
        Number of mel bins.
    fmin : Float64 >= 0 [scalar]
        Minimum frequency (Hz).
    fmax : Float64 >= 0 [scalar]
        Maximum frequency (Hz).
    htk : Bool
        If True, use HTK formula to convert Hz to mel.
        Otherwise (False), use Slaney's Auditory Toolbox.

    Returns
    -------
    bin_frequencies : ndarray [shape=(n_mels,)]
        Vector of ``n_mels`` frequencies in Hz which are uniformly spaced on the Mel
        axis.

    Examples
    --------
    >>> librosa.mel_frequencies(n_mels=40)
    array([     0.   ,     85.317,    170.635,    255.952,
              341.269,    426.586,    511.904,    597.221,
              682.538,    767.855,    853.173,    938.49 ,
             1024.856,   1119.114,   1222.042,   1334.436,
             1457.167,   1591.187,   1737.532,   1897.337,
             2071.84 ,   2262.393,   2470.47 ,   2697.686,
             2945.799,   3216.731,   3512.582,   3835.643,
             4188.417,   4573.636,   4994.285,   5453.621,
             5955.205,   6502.92 ,   7101.009,   7754.107,
             8467.272,   9246.028,  10096.408,  11025.   ])

    """
    # 'Center freqs' of mel bands - uniformly spaced between limits
    # https://github.com/librosa/librosa/blob/e403272fc984bc4aeb316e5f15899042224bb9fe/librosa/core/convert.py#L1648
    # [TODO] test against: librosa.mel_frequencies(n_mels=40)
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    hz: List[Float64] = mel_to_hz(mels, htk=htk)
    return hz

def hz_to_mel(
    frequencies: _ScalarOrSequence[_FloatLike_co], *, htk: bool = False
) -> Union[np.floating[Any], np.ndarray]:
# https://github.com/librosa/librosa/blob/e403272fc984bc4aeb316e5f15899042224bb9fe/librosa/core/convert.py#L1180C1-L1234C16
    """Convert Hz to Mels

    Examples
    --------
    >>> librosa.hz_to_mel(60)
    0.9
    >>> librosa.hz_to_mel([110, 220, 440])
    array([ 1.65,  3.3 ,  6.6 ])

    Parameters
    ----------
    frequencies : number or np.ndarray [shape=(n,)] , float
        scalar or array of frequencies
    htk : bool
        use HTK formula instead of Slaney

    Returns
    -------
    mels : number or np.ndarray [shape=(n,)]
        input frequencies in Mels

    See Also
    --------
    mel_to_hz
    """
    frequencies = np.asanyarray(frequencies)

    if htk:
        mels: np.ndarray = 2595.0 * np.log10(1.0 + frequencies / 700.0)
        return mels

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels

def mel_to_hz(
    mels: _ScalarOrSequence[_FloatLike_co], *, htk: bool = False
) -> Union[np.floating[Any], np.ndarray]:
# https://github.com/librosa/librosa/blob/e403272fc984bc4aeb316e5f15899042224bb9fe/librosa/core/convert.py#L1254C1-L1307C1
    """Convert mel bin numbers to frequencies

    Examples
    --------
    >>> librosa.mel_to_hz(3)
    200.

    >>> librosa.mel_to_hz([1,2,3,4,5])
    array([  66.667,  133.333,  200.   ,  266.667,  333.333])

    Parameters
    ----------
    mels : np.ndarray [shape=(n,)], float
        mel bins to convert
    htk : bool
        use HTK formula instead of Slaney

    Returns
    -------
    frequencies : np.ndarray [shape=(n,)]
        input mels in Hz

    See Also
    --------
    hz_to_mel
    """
    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


struct MelBands[num_bands: Int = 40, min_freq: Float64 = 20.0, max_freq: Float64 = 20000.0, fft_size: Int = 1024, htk: Bool = False](FFTProcessable):
    """Mel Bands analysis.

    Parameters:
        num_bands: The number of mel bands to compute.
        min_freq: The minimum frequency (in Hz) to consider when computing the mel bands.
        max_freq: The maximum frequency (in Hz) to consider when computing the mel bands.
        fft_size: The size of the FFT used to compute the mel bands.
    """

    var world: UnsafePointer[MMMWorld]

    fn __init__(out self, world: UnsafePointer[MMMWorld]):
        self.world = world

        # https://librosa.org/doc/main/generated/librosa.filters.mel.html
        # https://github.com/librosa/librosa/blob/e403272fc984bc4aeb316e5f15899042224bb9fe/librosa/filters.py#L128

        # Initialize the weights with zeros
        weights = List[List[Float64]](length=num_bands,fill=List[Float64](length=(fft_size // 2) + 1, fill=0.0))

        fftfreqs = fft_frequencies(sr=self.world[].sample_rate, n_fft=self.fft_size)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        mel_f = mel_frequencies(num_bands + 2, fmin=min_freq, fmax=max_freq, htk=htk)

        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fftfreqs)

        for i in range(num_bands):
            # lower and upper slopes for all bins
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]

            # .. then intersect them with each other and zero
            weights[i] = np.maximum(0, np.minimum(lower, upper))

        if isinstance(norm, str):
            if norm == "slaney":
                # Slaney-style mel is scaled to be approx constant energy per channel
                enorm = 2.0 / (mel_f[2 : num_bands + 2] - mel_f[:num_bands])
                weights *= enorm[:, np.newaxis]
            else:
                raise ParameterError(f"Unsupported norm={norm}")
        else:
            weights = util.normalize(weights, norm=norm, axis=-1)

        # Only check weights if f_mel[0] is positive
        if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
            # This means we have an empty channel somewhere
            warnings.warn(
                "Empty filters detected in mel frequency basis. "
                "Some channels will produce empty responses. "
                "Try increasing your sampling rate (and fmax) or "
                "reducing num_bands.",
                stacklevel=2,
            )

        # return weights

    fn next_frame(mut self, mut mags: List[Float64], mut phases: List[Float64]) -> None:
        """Compute the mel bands for a given FFT analysis.

        This function is to be used by FFTProcess if MelBands is passed as the "process".

        Args:
            mags: The input magnitudes as a List of Float64.
            phases: The input phases as a List of Float64.
        """
        pass # placeholder