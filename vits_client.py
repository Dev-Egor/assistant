import sys; sys.path.append("vits")
from torch import LongTensor, no_grad
from vits import commons
from vits import utils
from vits.models import SynthesizerTrn
from vits.text.symbols import symbols
from vits.text import text_to_sequence
import soundfile


hps = utils.get_hparams_from_file("personas/ayaka/tts/config.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint("personas/ayaka/tts/model.pth", net_g, None)

text_norm = text_to_sequence("こんにちは", hps.data.text_cleaners)
if hps.data.add_blank:
    text_norm = commons.intersperse(text_norm, 0)
stn_tst = LongTensor(text_norm)

with no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = LongTensor([stn_tst.size(0)]).cuda()
    sid = LongTensor([303]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.6, noise_scale_w=0.668, length_scale=1)[0][0, 0].data.cpu().float().numpy()
    soundfile.write("ayaka.wav", audio, 22050)