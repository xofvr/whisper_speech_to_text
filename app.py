import whisper
import os
import pyaudio
import wave

# Main function
def main(output_txt_path):
    # Load Whisper model
    model = whisper.load_model("base")
    
    # Access the microphone
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    
    # Start recording
    frames = []
    print("Recording started. Press Ctrl+C to stop recording.")
    try:
        while True:
            data = stream.read(1024)
            frames.append(data)
    except KeyboardInterrupt:
        pass
    
    # Stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Save the recorded audio to a WAV file
    wav_path = "recorded_audio.wav"
    with wave.open(wav_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wav_file.setframerate(16000)
        wav_file.writeframes(b''.join(frames))
    
    # Transcribe the speech to text
    transcription = model.transcribe(wav_path)
    
    # Save the transcription to a text file
    with open(output_txt_path, 'w') as txt_file:
        txt_file.write(transcription["text"])
    
    print("Transcription saved to:", output_txt_path)
    
    os.remove(wav_path)
    

if __name__ == "__main__":
    output_txt_path = "transcription.txt"
    main(output_txt_path)