import torch

print("=== PyTorch 라이브러리 정보 ===")
print(f"- PyTorch 버전: {torch.__version__}")
print(f"- CUDA 사용 가능 여부 (GPU): {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"- CUDA 버전: {torch.version.cuda}")
    print(f"- 현재 GPU 이름: {torch.cuda.get_device_name(0)}")
    # 670만개 데이터 학습 시 중요함 (VRAM 용량 확인)
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"- GPU 전체 메모리: {vram_total:.2f} GB")
else:
    print("⚠️ 경고: GPU를 사용할 수 없습니다. 이대로 진행하면 670만개 학습에 며칠이 걸릴 수 있습니다.")