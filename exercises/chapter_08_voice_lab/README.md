# Chapter 08 Voice Lab — Speech-to-Text and Text-to-Speech

This mini-lab documents two basic voice AI workflows in Python:

1. Speech-to-Text with `SpeechRecognition`, `PyAudio` and Google's Web Speech API.
2. Text-to-Speech with `pyttsx3` and the local Windows speech engine.

The goal is to reproduce the tutorial tasks in a small, documented lab inside the `ai-development-velptec` repository.

---

## Scope

This lab covers:

- listing available microphone devices,
- recording one spoken phrase,
- transcribing speech to text,
- handling speech recognition errors,
- running a continuous transcription loop,
- testing local text-to-speech output,
- documenting observed platform-specific behavior on Windows.

---

## Files

```text
exercises/chapter_08_voice_lab/
├── README.md
├── requirements-voice.txt
├── speech_to_text_demo.py
├── text_to_speech_demo.py
├── audio/
├── output/
└── reports/
```

---

## Requirements

The Python dependencies are listed in:

```text
requirements-voice.txt
```

Install them from the activated virtual environment:

```powershell
cd C:\dev\ai-development-velptec
.\.venv\Scripts\Activate.ps1

python -m pip install -r .\exercises\chapter_08_voice_lab\requirements-voice.txt
```

The lab was tested with:

```text
Python 3.11
Windows
FFmpeg 8.1.2
SpeechRecognition
PyAudio
pyttsx3
```

---

## Speech-to-Text

The Speech-to-Text demo uses:

```text
SpeechRecognition
PyAudio
Google Web Speech API
```

Important notes:

- Internet access is required for `recognize_google()`.
- No Google API key is required for low-volume tutorial usage.
- Audio is sent to Google's Web Speech API for transcription.
- The demo should not be used with private or sensitive information.

Run the demo:

```powershell
cd C:\dev\ai-development-velptec
.\.venv\Scripts\Activate.ps1

python .\exercises\chapter_08_voice_lab\speech_to_text_demo.py
```

The script performs:

1. microphone listing,
2. single-shot transcription,
3. continuous transcription,
4. timeout handling,
5. unknown-audio handling,
6. clean exit when the user says `quit`, `stop` or `exit`.

---

## Speech-to-Text Test Result

The Speech-to-Text demo was tested successfully.

### Detected microphones

The script detected multiple audio devices, including the internal microphone array:

```text
Mikrofonarray (Intel Smart Sound Technologie für digitale Mikrofone)
```

Some device names were displayed with encoding artifacts in the terminal, for example:

```text
Mikrofonarray (IntelÂ® Smart Sou...)
```

This did not prevent the microphone from working.

### Single-shot mode

Input spoken by the user:

```text
Hallo, ich teste die Spracherkennung mit Python.
```

Recognized output:

```text
hallo ich teste die Spracherkennung mit pied
```

Observation:

The transcription worked, but the word `Python` was incorrectly recognized as `pied`. This is an expected limitation of speech recognition systems, especially with technical terms, accents or background noise.

### Continuous mode

Recognized phrases:

```text
das ist ein zweiter Test
ihr funktioniert sehr gut
```

The following cases were also handled correctly:

```text
(no speech detected, still listening...)
(could not understand)
```

The loop stopped cleanly after the user said:

```text
exit
```

Final result:

```text
Speech-to-Text works successfully.
```

---

## Text-to-Speech

The Text-to-Speech demo uses:

```text
pyttsx3
Windows SAPI5 voices
```

The system detected the following voices:

```text
[0] Microsoft Hedda Desktop - German
[1] Microsoft David Desktop - English (United States)
[2] Microsoft Zira Desktop - English (United States)
```

The first spoken sentence worked successfully:

```text
Hello! This is a text to speech demonstration with Python.
```

However, the continuous speech loop showed a Windows-specific issue:

- the script continued running,
- each text input was printed correctly,
- the first sentence was spoken,
- later loop inputs were not spoken aloud.

Observed example:

```text
Text to speak: Hello, this is the first loop test.
Speaking: Hello, this is the first loop test.

Text to speak: Hello, this is the second loop test.
Speaking: Hello, this is the second loop test.
```

Only the first phrase was audible.

Current interpretation:

This appears to be a platform-specific behavior of `pyttsx3` with Windows SAPI5. The script logic works, but the audio output does not behave reliably in the continuous loop.

Planned follow-up:

If needed, this behavior will be documented in a GitHub issue or discussed with the instructor.

---

## Known Limitations

### Speech-to-Text

- Requires internet access.
- Recognition accuracy depends on microphone quality, pronunciation, background noise and the selected language code.
- Technical terms such as `Python` may be transcribed incorrectly.
- Audio is processed through Google's Web Speech API.

### Text-to-Speech

- `pyttsx3` works for single-shot speech on this Windows setup.
- The continuous loop currently does not speak every phrase reliably.
- The issue seems related to `pyttsx3` / Windows SAPI5 behavior rather than the loop logic itself.

---

## Conclusion

The Speech-to-Text part of the tutorial was completed successfully.

The Text-to-Speech part works partially:

- voice listing works,
- single-shot speech works,
- continuous loop logic works,
- repeated audio output in the loop is unreliable on this Windows environment.

This lab is useful as a practical introduction to voice interfaces and also shows why platform-specific testing is important for audio applications.