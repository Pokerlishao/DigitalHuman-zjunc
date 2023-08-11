cd /tts/monotonic_align
mkdir monotonic_align
python3 setup.py build_ext --inplace
cd ../..

wget https://huggingface.co/colin1639/VITS-fast-fine-tuning/blob/main/G_latest.pth -O ./tts/checkpoint/G_latest.pth