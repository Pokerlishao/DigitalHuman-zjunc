import sys
sys.path.append('./tts')
import torch
from torch import no_grad, LongTensor
import commons
import utils
from .models import SynthesizerTrn
from scipy.io.wavfile import write
from .text import text_to_sequence
device = "cuda:0" if torch.cuda.is_available() else "cpu"
import logging
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}
lang = ['日本語', '简体中文', 'English', 'Mix']

def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

class TTSWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config_path = './tts/config/finetune_speaker.json'
        model_path = './tts/checkpoint/G_latest.pth'
        self.hps = utils.get_hparams_from_file(config_path)
        
        self.model = SynthesizerTrn(
            len(self.hps.symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model).to(device).eval()
        utils.load_checkpoint(model_path, self.model, None)
        self.speaker_ids = self.hps.speakers
        
    def forward(self, text: str, language='简体中文', speaker='common', speed=1.0):
        text = language_marks[language] + text + language_marks[language]
        stn_tst = get_text(text, self.hps, False)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([self.speaker_ids[speaker]]).to(device)
            audio = self.model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        
        return self.hps.data.sampling_rate, audio
        
if __name__ == '__main__':
    tts = TTSWrapper()
    sampling_rate, audio = tts('随着科技的飞速发展，互联网已经深刻地改变了人们的生活方式和社会结构。信息的传播变得更加迅速和广泛，人们可以通过网络轻松获取各种知识和娱乐。社交媒体的兴起使人们能够跨越地域限制，与朋友、家人甚至陌生人保持联系。电子商务的兴盛使得购物变得更加便捷，人们可以足不出户就能购买各种商品和服务。然而，互联网的发展也带来了一些问题，如网络安全威胁、虚假信息传播等。因此，保障网络环境的安全和健康变得尤为重要。未来，随着技术的不断创新，我们可以期待互联网在更多领域发挥积极作用，同时也需要智慧地应对其中的挑战。', speed=1.0)
    write('output.wav', sampling_rate, audio)