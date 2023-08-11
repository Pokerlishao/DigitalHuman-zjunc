import whisper
from opencc import OpenCC
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class STTWrapper():
    def __init__(self):
        self.model = whisper.load_model('medium', device=device, download_root='./stt/checkpoint/')
        self.cc = OpenCC('t2s')
        
    def __call__(self, audio_path:str):
        result = self.model.transcribe(audio_path)
        return self.cc.convert(result['text'])

if __name__ == '__main__':
    stt = STTWrapper()
    text = stt('./output.wav')
    print(f'speech to text: {text}')