
with open('AddedRiemannFeatures.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = 'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # force CPU -- avoids CUDA kernel image mismatch on Kaggle'
if old not in content:
    old = 'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")'

new = '''# Safely test CUDA -- Kaggle PyTorch/driver versions sometimes mismatch
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    try:
        torch.zeros(1, device="cuda")   # quick kernel smoke-test
        DEVICE = torch.device("cuda")
        print(f"[MiniLCM] GPU: {torch.cuda.get_device_name(0)}")
    except Exception as _cuda_err:
        print(f"[MiniLCM] CUDA failed ({_cuda_err}) -- using CPU")'''

assert old in content, f"anchor not found"
content = content.replace(old, new, 1)

with open('AddedRiemannFeatures.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("done")
