import numpy as np
import soundfile as sf
from dataclasses import dataclass
from enum import Enum
import argparse

class Quality(Enum):
    kCoarse = 0
    kRefined = 1

@dataclass()
class LagEstimate:
    accuracy: float
    reliable: bool
    lag: int
    updated: bool

@dataclass()
class render_buffer:
    size: int
    write: int
    read: int
    buffer: np.array

class timeDelayEstimate:
    def __init__(self):
        self.error_sum = 0
        self.smoothing = 0.7
        self.default_block_size = 64
        self.down_sample_rate = 4
        self.sub_block_size_ = self.default_block_size // self.down_sample_rate
        self.window_size_sub_blocks = 32
        self.overlap_size_sub_blocks = 8
        self.nb_nlms = 5
        self.sp_nlms = self.window_size_sub_blocks * self.sub_block_size_
        self.render_buffer_size = self.sub_block_size_ * ((self.window_size_sub_blocks - self.overlap_size_sub_blocks) * self.nb_nlms) \
                                  + self.window_size_sub_blocks + 1
        self.kNumBlocksPerSecond = 250
        self.filter_ = np.zeros((self.nb_nlms, self.sp_nlms))
        self.excitation_limit_ = 150
        self.filter_intra_lag_shift_ = (self.window_size_sub_blocks - self.overlap_size_sub_blocks) * self.sub_block_size_
        self.matching_filter_threshold_ = 0.2
        self.lag_estimates_ = []
        for i in range(self.nb_nlms):
            self.lag_estimates_.append(LagEstimate(0,  False, 0, False))
        self.histogram_ = np.zeros(self.nb_nlms * self.filter_intra_lag_shift_ + self.window_size_sub_blocks * self.sub_block_size_ + 1)
        self.histogram_data_ = np.zeros(self.kNumBlocksPerSecond)
        self.histogram_data_index_ = 0
        self.render_buffer = np.zeros(2448)
        self.render_buffer_write_index = 0
        self.render_buffer_read_index = 0


        # 滤波器系数
        self.b1 = np.array([0.972609997, -1.94523001, 0.972609997])
        self.a1 = np.array([-1.94447994, 0.945980012])
        self.b_m = np.array(
            [[0.262507, 0.0465889, 0.262507], [0.262507, -0.326946, 0.262507], [0.262507, -0.373325, 0.262507]])
        self.a_m = np.array([[-1.518325, 0.633167], [-1.49784, 0.85358], [-1.4979, 0.96957]])
        self.b2 = np.array([0.757076, -1.514153, 0.757076])
        self.a2 = np.array([-1.454244, 0.574062])
        # 数据缓存
        self.sig_m_x_1 = np.zeros(2)
        self.sig_m_y_1 = np.zeros(2)
        self.sig_m_x_m = np.zeros([3, 2])
        self.sig_m_y_m = np.zeros([3, 2])
        self.sig_m_x_2 = np.zeros(2)
        self.sig_m_y_2 = np.zeros(2)

        self.ref_m_x_1 = np.zeros(2)
        self.ref_m_y_1 = np.zeros(2)
        self.ref_m_x_m = np.zeros([3,2])
        self.ref_m_y_m = np.zeros([3,2])
        self.ref_m_x_2 = np.zeros(2)
        self.ref_m_y_2 = np.zeros(2)

        #
        self.buffer_headroom_ = 12 # 滤波器个数
        self.block_buffer_size = 187 # 缓存的block数
        self.delay_ = None
        self.consistent_estimate_counter_ = 0
        self.old_aggregated_lag_ = None
        self.old_aggregated_lag_delay = 0
        self.delay_change_counter_ = 0
        self.delay_samples_ = None
        self.delay_samples_delay = 0
        self.last_delay_estimate_quality_ = Quality.kCoarse
        self.delay_samples_quality = Quality.kCoarse
        self.delay_headroom_samples = 33
        #
        self.thresholds_converged_ = 20
        self.thresholds_initial = 5
        self.capture_properly_started_ = False
        self.filters_updated = False
        self.significant_candidate_found_ = False

    def BufferLatency(self):
        latency_samples = (self.render_buffer.shape[0] + self.render_buffer_read_index - self.render_buffer_write_index) % self.render_buffer.shape[0]
        latency_blocks = latency_samples / self.sub_block_size_
        return latency_blocks

    def MapDelayToTotalDelay(self, external_delay_blocks):
        self.delay_ = external_delay_blocks
        return min(self.block_buffer_size - self.buffer_headroom_ - 1, max(self.BufferLatency() + external_delay_blocks,0))

    def AlignFromDelay(self, delay):
        # 这里省略了，AEC3中就要把这个totaldelay应用到Buffer Block中，方法即为减对应的block即可
        if self.delay_ != delay:
            print(self.MapDelayToTotalDelay())

    def DownSample(self, capture, down_sample_rate):
        downcapture = np.zeros(capture.shape[0] // down_sample_rate)
        for i in range(capture.shape[0] // down_sample_rate):
            downcapture[i] = capture[i * down_sample_rate]
        return downcapture

    # 高通滤波, 近端噪声抑制滤波器
    def cascaded_biquad_filter(self, wav, b, a, m_x, m_y):
        # b = np.array([0.97261, -1.94523, 0.97261])
        # a = np.array([-1.94448, 0.94598])
        wav_out = np.zeros_like(wav)
        for i in range(wav.shape[0]):
            tmp = wav[i]
            wav_out[i] = b[0] * tmp + b[1] * m_x[0] + b[2] * m_x[1] \
                         - a[0] * m_y[0] - a[1] * m_y[1]
            m_x[1] = m_x[0]
            m_x[0] = tmp
            m_y[1] = m_y[0]
            m_y[0] = wav_out[i]
        return wav_out, m_x, m_y

    # 多滤波器频带抑制
    def cascaded_biquad_multi_filter(self, wav, b, a, m_x, m_y):
        for j in range(b.shape[0]):
            wav_out = np.zeros_like(wav)
            for i in range(wav.shape[0]):
                tmp = wav[i]
                wav_out[i] = b[j][0] * tmp + b[j][1] * m_x[j][0] + b[j][2] * m_x[j][1] \
                             - a[j][0] * m_y[j][0] - a[j][1] * m_y[j][1]
                m_x[j][1] = m_x[j][0]
                m_x[j][0] = tmp
                m_y[j][1] = m_y[j][0]
                m_y[j][0] = wav_out[i]
            wav = wav_out
        return wav_out, m_x, m_y

    def MatchedFilterCore(self, x_start_index, x2_sum_threshold, sig, h):

        # 每一个点计算一遍，相当于卷积操作
        # 16点 64点  32 block
        for i in range(sig.shape[0]):
            x2_sum = 0
            s = 0
            x_index = x_start_index
            for k in range(self.sp_nlms):
                x2_sum += self.render_buffer[x_index] * self.render_buffer[x_index]
                s += h[k] * self.render_buffer[x_index]
                x_index = x_index + 1 if x_index < (self.render_buffer.shape[0] - 1) else 0
        # 计算匹配滤波器error
            e = sig[i] - s
            saturation = (sig[i] >= 32000 or sig[i] <= -32000)
            self.error_sum += e*e
            if x2_sum > x2_sum_threshold and (not saturation):
                self.filters_updated = True
                alpha = self.smoothing * e / x2_sum
                x_index = x_start_index
                for k in range(self.sp_nlms):
                    h[k] += alpha * self.render_buffer[x_index]
                    x_index = x_index + 1 if x_index < (self.render_buffer.shape[0] - 1) else 0

            x_start_index = x_start_index - 1 if x_start_index > 0 else self.render_buffer.shape[0] - 1
        return h

    def MatchedFilterLagAggregator(self):
        best_accuracy = 0
        best_lag_estimate_index = -1
        for k in range(len(self.lag_estimates_)):
            if self.lag_estimates_[k].updated and self.lag_estimates_[k].reliable:
                if self.lag_estimates_[k].accuracy > best_accuracy:
                    best_accuracy = self.lag_estimates_[k].accuracy
                    best_lag_estimate_index = k
        if best_lag_estimate_index != -1:
            self.histogram_[int(self.histogram_data_[self.histogram_data_index_])] -= 1
            self.histogram_data_[self.histogram_data_index_] = self.lag_estimates_[best_lag_estimate_index].lag
            self.histogram_[int(self.histogram_data_[self.histogram_data_index_])] += 1
            self.histogram_data_index_ = (self.histogram_data_index_ + 1) % self.histogram_data_.shape[0]
            candidate = np.argmax(self.histogram_)
            print(f"candidate:{candidate} histogram_[candidate]:{self.histogram_[candidate]}")
            self.significant_candidate_found_ = self.significant_candidate_found_ or (self.histogram_[candidate] > self.thresholds_converged_)
            if self.histogram_[candidate] > self.thresholds_converged_ or (self.histogram_[candidate] > self.thresholds_initial and not self.significant_candidate_found_):
                quality = Quality.kRefined if self.significant_candidate_found_ else Quality.kCoarse
                return candidate, quality

        return None, -1


    def Update(self, capture):
        y = capture
        x2_sum_threshold = self.filter_.shape[1] * self.excitation_limit_ ** 2
        alignment_shift = 0
        for n in range(self.filter_.shape[0]):
            self.error_sum = 0
            self.filters_updated = False
            x_start_index = (self.render_buffer_read_index + self.sub_block_size_ + alignment_shift - 1) % self.render_buffer.shape[0]
            self.filter_[n] = self.MatchedFilterCore(x_start_index, x2_sum_threshold,  y,
                                   self.filter_[n])
            error_sum_anchor = np.sum(y ** 2)
            lag_estimate = np.argmax(self.filter_[n] ** 2)
            flag = lag_estimate > 2 and lag_estimate < (self.filter_.shape[1] - 10) and self.error_sum < self.matching_filter_threshold_ * error_sum_anchor
            self.lag_estimates_[n] = LagEstimate(error_sum_anchor - self.error_sum,
                                                 flag,
                                                 lag_estimate + alignment_shift, self.filters_updated)
            if flag:
                print(f"accuracy:{round(error_sum_anchor - self.error_sum, 5)} reliable:{flag} lag:{lag_estimate + alignment_shift} "
                      f"updated:{self.filters_updated}")

            alignment_shift += self.filter_intra_lag_shift_

    def BufferReadReset(self):
        self.render_buffer_read_index = (self.render_buffer_write_index + self.sub_block_size_) % self.render_buffer.shape[0]

    def BufferReadMove(self):
        self.render_buffer_read_index = (self.render_buffer_read_index - self.sub_block_size_) % \
                                        self.render_buffer.shape[0]

    def SigFilterProcess(self, sig):
        sig, self.sig_m_x_1, self.sig_m_y_1 = self.cascaded_biquad_filter(sig, self.b1, self.a1, self.sig_m_x_1, self.sig_m_y_1)
        sig, self.sig_m_x_m, self.sig_m_y_m = self.cascaded_biquad_multi_filter(sig, self.b_m, self.a_m, self.sig_m_x_m, self.sig_m_y_m)
        sig, self.sig_m_x_2, self.sig_m_y_2 = self.cascaded_biquad_filter(sig, self.b2, self.a2, self.sig_m_x_2, self.sig_m_y_2)
        sig = self.DownSample(sig, self.down_sample_rate)
        return sig

    def RefFilterProcess(self, ref):
        ref, self.ref_m_x_1, self.ref_m_y_1 = self.cascaded_biquad_filter(ref, self.b1, self.a1, self.ref_m_x_1, self.ref_m_y_1)
        ref, self.ref_m_x_m, self.ref_m_y_m = self.cascaded_biquad_multi_filter(ref, self.b_m, self.a_m, self.ref_m_x_m, self.ref_m_y_m)
        ref, self.ref_m_x_2, self.ref_m_y_2 = self.cascaded_biquad_filter(ref, self.b2, self.a2, self.ref_m_x_2, self.ref_m_y_2)
        downref = self.DownSample(ref, self.down_sample_rate)
        self.render_buffer_write_index = (self.render_buffer_write_index - downref.shape[0]) % self.render_buffer.shape[0]
        self.render_buffer[self.render_buffer_write_index:self.render_buffer_write_index + downref.shape[0]] = downref[::-1]



    def EstimateDelay(self, downcapture):
        self.Update(downcapture)
        candidate, quality = self.MatchedFilterLagAggregator()

        if candidate is not None:
            delay = candidate * 4
            return delay

        return 0

    def HistogramReset(self):
        self.filter_ = np.zeros_like(self.filter_)
        for i in range(self.nb_nlms):
            self.lag_estimates_.append(LagEstimate(0, False, 0, False))
        self.old_aggregated_lag_ = None
        self.consistent_estimate_counter_ = 0

    def ComputeBufferDelay(self, hysteresis_limit_blocks):
        delay_with_headroom_samples = max(self.delay_samples_delay - self.delay_headroom_samples, 0)
        new_delay_blocks = delay_with_headroom_samples // 64
        if self.delay_ is not None:
            current_delay_blocks = self.delay_
            if new_delay_blocks > current_delay_blocks and new_delay_blocks <= current_delay_blocks + hysteresis_limit_blocks:
                new_delay_blocks = current_delay_blocks

        return new_delay_blocks


    def GetDelay(self, downcapture):
        # 这个和EstimateDelay有重复，为了补充AEC3 时延估计后续的一些细节操作
        self.Update(downcapture)
        candidate, quality = self.MatchedFilterLagAggregator()
        delay = 0
        if candidate is not None:
            delay = candidate * self.down_sample_rate
        if self.old_aggregated_lag_ is not None and candidate is not None and self.old_aggregated_lag_delay == candidate * self.down_sample_rate:
            self.consistent_estimate_counter_ += 1
        else:
            self.consistent_estimate_counter_ = 0
        self.old_aggregated_lag_ = candidate
        self.old_aggregated_lag_delay = candidate * self.down_sample_rate
        if self.consistent_estimate_counter_ > self.kNumBlocksPerSecond / 2:
            self.HistogramReset()
        delay_samples = candidate
        if delay_samples is not None:
            if self.delay_samples_ is None or (self.delay_samples_delay != delay):
                self.delay_change_counter_ = 0
            if self.delay_samples_delay is not None:
                # 参数没用，所以省略了
                print(f"modify delay_samples params")
            else:
                self.delay_samples_ = delay_samples
                self.delay_samples_delay = delay
                self.delay_samples_quality = quality
        else:
            if self.delay_samples_ is not None:
                # 参数没用，所以省略了
                print(f"modify delay_samples params")

        if self.delay_change_counter_ < 2 * self.kNumBlocksPerSecond:
            self.delay_change_counter_ += 1

        if self.delay_samples_ is not None:
            use_hysteresis = (self.last_delay_estimate_quality_ == Quality.kRefined and self.delay_samples_quality == Quality.kRefined)
            self.delay_ = self.ComputeBufferDelay(1 if use_hysteresis else 0)
            self.last_delay_estimate_quality_ = self.delay_samples_quality

        return self.delay_




def gcc_phat(sig, ref):

    tde = timeDelayEstimate()
    M = 64
    num_block = min(len(sig), len(ref)) // M
    timedelay = np.zeros(num_block)

    for n in range(num_block-1):
        if tde.capture_properly_started_ == False:
            tde.capture_properly_started_ = True
            y_n = ref[n * M:(n + 1) * M]
            x_n = sig[n * M:(n + 1) * M]
            tde.RefFilterProcess(y_n)
            n += 1
            y_n = ref[n * M:(n + 1) * M]
            tde.RefFilterProcess(y_n)
            tde.BufferReadReset()
            tde.BufferReadMove()
            x_n = tde.SigFilterProcess(x_n)
            timedelay[n-1] = tde.EstimateDelay(x_n)
            x_n = sig[n * M:(n + 1) * M]
            x_n = tde.SigFilterProcess(x_n)
            timedelay[n-1] = tde.EstimateDelay(x_n)

        else:
            n += 1
            y_n = ref[n * M:(n + 1) * M]
            tde.RefFilterProcess(y_n)
            tde.BufferReadMove()
            x_n = sig[n*M:(n+1)*M]
            x_n = tde.SigFilterProcess(x_n)
            timedelay[n-1] = tde.EstimateDelay(x_n)


    return timedelay

def get_parser():
    parser = argparse.ArgumentParser(description="data input")
    parser.add_argument(
        "--rec",
        dest='rec',
        type=str,
        required=False,
        default="./gcc_phat_wav/ic_mic.wav",
        help="mic signal",
    )
    parser.add_argument(
        "--ref",
        dest='ref',
        type=str,
        required=False,
        default="./gcc_phat_wav/ic_ref.wav",
        help="ref",
    )


    return parser

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()


    wav1, fs = sf.read(args.rec)
    wav2, fs = sf.read(args.ref)

    wav1 = wav1*32768
    wav2 = wav2*32768

    timedelay = gcc_phat(wav1, wav2)
    print(timedelay)

