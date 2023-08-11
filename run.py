from tts.tts_wrapper import TTSWrapper
from stt.stt_wrapper import STTWrapper
from scipy.io.wavfile import write

if __name__ == '__main__':
    # Test TTSWrapper
    tts = TTSWrapper()
    sampling_rate, audio = tts('随着科技的飞速发展，互联网已经深刻地改变了人们的生活方式和社会结构。信息的传播变得更加迅速和广泛，人们可以通过网络轻松获取各种知识和娱乐。社交媒体的兴起使人们能够跨越地域限制，与朋友、家人甚至陌生人保持联系。电子商务的兴盛使得购物变得更加便捷，人们可以足不出户就能购买各种商品和服务。然而，互联网的发展也带来了一些问题，如网络安全威胁、虚假信息传播等。因此，保障网络环境的安全和健康变得尤为重要。未来，随着技术的不断创新，我们可以期待互联网在更多领域发挥积极作用，同时也需要智慧地应对其中的挑战。', speed=1.0)
    write('./output.wav', sampling_rate, audio)
    
    # Test STTWrapper
    stt = STTWrapper()
    text = stt('./output.wav')
    print(f'speech to text: {text}')