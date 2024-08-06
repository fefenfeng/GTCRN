import os
import torch
import soundfile as sf
from gtcrn import GTCRN

mix_name = 'p232_005.wav'
enh_name = mix_name[:-4] + '_enh.wav'

# load model
device = torch.device("cpu")
model = GTCRN().eval()
ckpt = torch.load(os.path.join('checkpoints', 'model_trained_on_dns3.tar'), map_location=device)
model.load_state_dict(ckpt['model'])

# load data, data采样率需要为16kHz
mix, fs = sf.read(os.path.join('test_wavs', mix_name), dtype='float32')
assert fs == 16000

# inference，平方根汉宁窗
input = torch.stft(torch.from_numpy(mix), 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
with torch.no_grad():
    output = model(input[None])[0]
enh = torch.istft(output, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)

# save enhanced wav
sf.write(os.path.join('test_wavs', enh_name), enh.detach().cpu().numpy(), fs)
