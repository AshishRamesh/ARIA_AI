"""
Install:
    pip install --user kokoro soundfile sounddevice numpy
    sudo apt install espeak-ng

Run:
    python3 test_kokoro_tts.py
    python3 test_kokoro_tts.py "your custom sentence here"
    python3 test_kokoro_tts.py "hello" af_bella
    python3 test_kokoro_tts.py "hello" all      # audition every voice
"""
import sys
import time
import numpy as np
import sounddevice as sd
from kokoro import KPipeline

SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"
DEFAULT_TEXT = (
    "Hello, I am ARIA. I can navigate to the petrol pump, "
    "the burger king, the apartment area, or the parking lot. "
    "What would you like me to do?"
)

VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael",
    "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
]


def speak(pipeline, text, voice, stream):
    gen_start = time.time()
    first_audio_at = None
    total_samples = 0
    for _, _, audio in pipeline(text, voice=voice, speed=1.0):
        if first_audio_at is None:
            first_audio_at = time.time() - gen_start
            print(f"  first audio: {first_audio_at*1000:.0f} ms")
        audio = np.asarray(audio, dtype=np.float32)
        total_samples += len(audio)
        stream.write(audio)
    total = time.time() - gen_start
    audio_dur = total_samples / SAMPLE_RATE
    rtf = total / audio_dur if audio_dur else float("inf")
    print(f"  audio {audio_dur:.2f}s  wall {total:.2f}s  rtf {rtf:.2f}x")


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT
    voice_arg = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_VOICE

    print(f"Text: {text!r}")
    print()

    print("Loading Kokoro pipeline...")
    t0 = time.time()
    pipeline = KPipeline(lang_code="a")
    print(f"  loaded in {time.time() - t0:.2f}s\n")

    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    stream.start()

    voices = VOICES if voice_arg == "all" else [voice_arg]
    intro_text = "This is voice {name}."

    try:
        for voice in voices:
            print(f"=== {voice} ===")
            if voice_arg == "all":
                speak(pipeline, intro_text.format(name=voice.replace("_", " ")), voice, stream)
            speak(pipeline, text, voice, stream)
            print()
            if voice_arg == "all":
                time.sleep(0.5)
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
