import whisper
# model = whisper.load_model("base")
model = whisper.load_model("base", device="cuda")  # 使用 GPU
result = model.transcribe("recording.wav")
print(result["text"])