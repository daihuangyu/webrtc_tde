import soundfile as sf
import numpy as np


# 高通滤波, 近端噪声抑制滤波器
def cascaded_biquad_filter(wav, b, a):
    # b = np.array([0.97261, -1.94523, 0.97261])
    # a = np.array([-1.94448, 0.94598])
    wav_out = np.zeros_like(wav)
    m_x = np.zeros(2)
    m_y = np.zeros(2)
    for i in range(wav.shape[0]):
        tmp = wav[i]
        wav_out[i] = b[0] * tmp + b[1] * m_x[0] + b[2] * m_x[1] \
                     - a[0] * m_y[0] - a[1] * m_y[1]
        m_x[1] = m_x[0]
        m_x[0] = tmp
        m_y[1] = m_y[0]
        m_y[0] = wav_out[i]
    return wav_out

# 多滤波器频带抑制
def cascaded_biquad_multi_filter(wav, b, a):
    for j in range(b.shape[0]):
        wav_out = np.zeros_like(wav)
        m_x = np.zeros(2)
        m_y = np.zeros(2)
        for i in range(wav.shape[0]):
            tmp = wav[i]
            wav_out[i] = b[j][0] * tmp + b[j][1] * m_x[0] + b[j][2] * m_x[1] \
                         - a[j][0] * m_y[0] - a[j][1] * m_y[1]
            m_x[1] = m_x[0]
            m_x[0] = tmp
            m_y[1] = m_y[0]
            m_y[0] = wav_out[i]
        wav = wav_out
    return wav

# 下采样
def down_sample(capture, down_sample_rate):
    downcapture = np.zeros(capture.shape[0] // down_sample_rate)
    for i in range(capture.shape[0] // down_sample_rate):
        downcapture[i] = capture[i * down_sample_rate]
    return downcapture


if __name__ == '__main__':
    wav1, fs = sf.read("./gcc_phat_wav/rec.wav")
    wav2, fs = sf.read("./gcc_phat_wav/ref.wav")

    wav1 = wav1*32768
    wav2 = wav2*32768

    ref = wav2[:128]
    render_buffer = np.zeros(128)
    b1 = np.array([0.97261, -1.94523, 0.97261])
    a1 = np.array([-1.94448, 0.94598])
    b_m = np.array([[0.262507, 0.0465889, 0.262507], [0.262507, -0.326946, 0.262507], [0.262507, -0.373325, 0.262507]])
    a_m = np.array([[-1.518325, 0.633167], [-1.49784, 0.85358], [-1.4979, 0.96957]])
    b2 = np.array([0.757076, -1.514153, 0.757076])
    a2 = np.array([-1.454244, 0.574062])
    for i in range(2):
        x_n = ref[i*64:(i+1)*64]
        x_n_filter1 = cascaded_biquad_filter(x_n, b1, a1)
        x_n_filter2 = cascaded_biquad_multi_filter(x_n_filter1, b_m, a_m)
        x_n_filter3 = cascaded_biquad_filter(x_n_filter2, b2, a2)
        x_n_out = down_sample(x_n_filter3, 4)
        print(x_n_out)





