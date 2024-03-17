import soundfile as sf
import numpy as np
import os
from pydub import AudioSegment

def show_pcm4(file_names, output_name):

    wav1 = AudioSegment.from_file(f'{file_names[0]}', format="wav")
    wav2 = AudioSegment.from_file(f'{file_names[1]}', format="wav")
    wav3 = AudioSegment.from_file(f'{file_names[2]}', format="wav")
    wav4 = AudioSegment.from_file(f'{file_names[3]}', format="wav")
    wav5 = AudioSegment.from_file(f'{file_names[4]}', format="wav")
    wav6 = AudioSegment.from_file(f'{file_names[5]}', format="wav")
    wav1.export(f'{file_names[0].split(".wav")[0]}.pcm', format="raw")
    wav2.export(f'{file_names[1].split(".wav")[0]}.pcm', format="raw")
    wav3.export(f'{file_names[2].split(".wav")[0]}.pcm', format="raw")
    wav4.export(f'{file_names[3].split(".wav")[0]}.pcm', format="raw")
    wav5.export(f'{file_names[4].split(".wav")[0]}.pcm', format="raw")
    wav6.export(f'{file_names[5].split(".wav")[0]}.pcm', format="raw")
    pcm_1 = AudioSegment.from_file(f'{file_names[0].split(".wav")[0]}.pcm', sample_width=2, frame_rate=16000, channels=1)
    pcm_2 = AudioSegment.from_file(f'{file_names[1].split(".wav")[0]}.pcm', sample_width=2, frame_rate=16000, channels=1)
    pcm_3 = AudioSegment.from_file(f'{file_names[2].split(".wav")[0]}.pcm', sample_width=2, frame_rate=16000, channels=1)
    pcm_4 = AudioSegment.from_file(f'{file_names[3].split(".wav")[0]}.pcm', sample_width=2, frame_rate=16000, channels=1)
    pcm_5 = AudioSegment.from_file(f'{file_names[4].split(".wav")[0]}.pcm', sample_width=2, frame_rate=16000, channels=1)
    pcm_6 = AudioSegment.from_file(f'{file_names[5].split(".wav")[0]}.pcm', sample_width=2, frame_rate=16000, channels=1)
    stereo_sound = AudioSegment.from_mono_audiosegments(pcm_1, pcm_2, pcm_3, pcm_4, pcm_5, pcm_6)
    stereo_sound.export(f'{output_name}', format='raw')
    os.remove(f'{file_names[0].split(".wav")[0]}.pcm')
    os.remove(f'{file_names[1].split(".wav")[0]}.pcm')
    os.remove(f'{file_names[2].split(".wav")[0]}.pcm')
    os.remove(f'{file_names[3].split(".wav")[0]}.pcm')
    os.remove(f'{file_names[4].split(".wav")[0]}.pcm')
    os.remove(f'{file_names[5].split(".wav")[0]}.pcm')

if __name__ == '__main__':
    # wav1, fs = sf.read("./gcc_phat_wav/icm_mic.wav")
    # wav2 = np.concatenate((np.random.uniform(-0.01, 0.01, size=4000), wav1[:]), axis=0)
    # sf.write("./gcc_phat_wav/ic_mic.wav", wav2, samplerate=16000)

    wav1, fs = sf.read("./gcc_phat_wav/icm_mic.wav")
    wav2, fs = sf.read("./gcc_phat_wav/ic_ref.wav")
    #wav1 = wav1[:wav2.shape[0]]
    wav5 = wav1
    wav6 = wav2
    sf.write("./gcc_phat_wav/ic_mic.wav", wav1, samplerate=16000)
    sf.write("./gcc_phat_wav/ic_mic2.wav", wav5, samplerate=16000)
    sf.write("./gcc_phat_wav/ic_ref2.wav", wav6, samplerate=16000)
    wav3 = np.zeros_like(wav1)
    wav4 = np.zeros_like(wav1)
    sf.write("./gcc_phat_wav/ic_blank1.wav", wav3, samplerate=16000)
    sf.write("./gcc_phat_wav/ic_blank2.wav", wav4, samplerate=16000)
    filenames = ["./gcc_phat_wav/ic_mic.wav", "./gcc_phat_wav/ic_mic2.wav", "./gcc_phat_wav/ic_blank1.wav",
                 "./gcc_phat_wav/ic_blank2.wav", "./gcc_phat_wav/ic_ref.wav", "./gcc_phat_wav/ic_ref2.wav"]
    show_pcm4(filenames, "./gcc_phat_wav/out_nmf.pcm")



