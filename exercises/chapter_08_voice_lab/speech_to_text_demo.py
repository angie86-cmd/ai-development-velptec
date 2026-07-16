import speech_recognition as sr


# Google Speech-to-Text tutorial with Python.
# This script uses the SpeechRecognition library and Google's Web Speech API.
# The microphone input is handled through PyAudio.
#
# Important:
# - Internet access is required for recognize_google().
# - No Google API key is required for low-volume tutorial usage.
# - Do not use this demo with private or sensitive information.
#
# Tutorial coverage:
# - Task 1: list available microphones.
# - Task 2: run one single-shot transcription.
# - Task 3: handle UnknownValueError and RequestError separately.
# - Task 4: run a continuous transcription loop.
# - Task 5: combine single-shot mode and continuous mode in __main__.


def list_microphones():
    # Print all microphone devices detected by SpeechRecognition / PyAudio.
    # The index shown here can be passed to sr.Microphone(device_index=...).
    print("Available microphones:")

    microphones = sr.Microphone.list_microphone_names()

    if not microphones:
        print("  No microphones were detected.")
        return

    for index, name in enumerate(microphones):
        print(f"  [{index}] {name}")


def transcribe_once(mic_index=None, language="en-US"):
    # Create a recognizer instance.
    recognizer = sr.Recognizer()

    # Open either the default microphone or a specific microphone by index.
    mic = sr.Microphone(device_index=mic_index)

    try:
        with mic as source:
            print("Calibrating for ambient noise... please stay quiet.")

            # Calibrate once before recording.
            # This helps the recognizer distinguish speech from background noise.
            recognizer.adjust_for_ambient_noise(source, duration=1)

            print("Listening... please speak now.")

            # Capture one spoken phrase.
            audio = recognizer.listen(source)

        print("Transcribing...")

        # Send the captured audio to Google's Web Speech API.
        text = recognizer.recognize_google(audio, language=language)

        return text

    except sr.UnknownValueError:
        # This means audio was captured, but Google could not understand it.
        print("Could not understand audio.")
        return None

    except sr.RequestError as error:
        # This means the API request failed.
        # Possible causes: no internet connection, service unavailable, quota issue.
        print(f"Google API request failed: {error}")
        return None

    except OSError as error:
        # This can happen if the microphone is unavailable, blocked or misconfigured.
        print(f"Microphone error: {error}")
        return None


def transcribe_loop(mic_index=None, language="en-US"):
    # Create one recognizer instance for the complete loop.
    recognizer = sr.Recognizer()

    # Open either the default microphone or a specific microphone by index.
    mic = sr.Microphone(device_index=mic_index)

    try:
        # Calibrate for ambient noise once before entering the loop.
        # Recalibrating before every phrase would slow down the interaction.
        with mic as source:
            print("Calibrating for ambient noise once before the loop...")
            recognizer.adjust_for_ambient_noise(source, duration=1)

    except OSError as error:
        print(f"Microphone error during calibration: {error}")
        return

    print("\nListening continuously.")
    print("Say 'quit', 'stop' or 'exit' to end the loop.\n")

    while True:
        try:
            with mic as source:
                print("Listening...")

                # timeout=5:
                # If no speech starts within 5 seconds, WaitTimeoutError is raised.
                #
                # phrase_time_limit=10:
                # Each spoken phrase is capped at 10 seconds.
                audio = recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=10
                )

        except sr.WaitTimeoutError:
            # No speech started within 5 seconds.
            # The loop continues instead of crashing.
            print("(no speech detected, still listening...)")
            continue

        except OSError as error:
            # Microphone became unavailable during the loop.
            print(f"Microphone error while listening: {error}")
            break

        try:
            # Send the phrase to Google's Web Speech API.
            text = recognizer.recognize_google(audio, language=language)

            print(f"You said: {text}")

            # Stop words are checked case-insensitively.
            if text.lower() in ("quit", "stop", "exit"):
                print("Stopping.")
                break

        except sr.UnknownValueError:
            # Speech was detected, but could not be understood.
            # This is recoverable, so the loop continues.
            print("(could not understand)")
            continue

        except sr.RequestError as error:
            # API/network errors are not usually fixed by immediate retry.
            # The tutorial requires breaking out of the loop on API error.
            print(f"API error: {error}")
            break


def main():
    print("=== Google Speech-to-Text Tutorial ===\n")

    # Configuration for this local demo.
    #
    # mic_index:
    # - None means the system default microphone.
    # - To use a specific microphone, first check the printed list below
    #   and replace None with the desired index, for example: 1.
    #
    # language:
    # - "en-US" for English.
    # - "de-DE" for German.
    # - "es-ES" for Spanish.
    mic_index = None
    language = "de-DE"

    # Task 1: show all available microphones.
    list_microphones()

    # Task 2: run one single-shot transcription.
    print("\n--- Single-shot mode ---")
    result = transcribe_once(
        mic_index=mic_index,
        language=language
    )

    if result:
        print(f"Transcription: {result}\n")
    else:
        print("No transcription was returned.\n")

    # Task 4 and Task 5: start continuous transcription after single-shot mode.
    print("--- Continuous mode ---")
    transcribe_loop(
        mic_index=mic_index,
        language=language
    )


if __name__ == "__main__":
    main()