import pyttsx3


# Text-to-Speech demo with pyttsx3.
# This version focuses on the continuous speech loop.
# It avoids speaking a demo sentence before the loop, because on some Windows
# systems pyttsx3 / SAPI5 only speaks the first queued sentence reliably.


def list_voices(engine):
    # Get all voices available on the current system.
    voices = engine.getProperty("voices")

    print("Available voices:")

    # Print each voice with an index.
    for index, voice in enumerate(voices):
        print(f"[{index}] {voice.name} - {voice.id}")


def configure_engine(engine, voice_index=None, rate=150, volume=1.0):
    # Set speech rate.
    engine.setProperty("rate", rate)

    # Set volume.
    engine.setProperty("volume", volume)

    # Optionally select a specific voice.
    if voice_index is not None:
        voices = engine.getProperty("voices")

        if 0 <= voice_index < len(voices):
            engine.setProperty("voice", voices[voice_index].id)
            print(f"Selected voice: {voices[voice_index].name}")
        else:
            print(f"Voice index {voice_index} is not valid. Using default voice.")


def speak_loop(engine):
    # Reuse one engine instance during the loop.
    print("\nText-to-Speech loop.")
    print("Type a sentence and press Enter.")
    print("Type 'quit', 'stop' or 'exit' to end the loop.\n")

    while True:
        # Ask the user for text input.
        user_text = input("Text to speak: ").strip()

        # Stop cleanly when the user enters an exit command.
        if user_text.lower() in ["quit", "stop", "exit"]:
            print("Stopping Text-to-Speech loop.")
            break

        # Skip empty input.
        if not user_text:
            print("Empty input skipped. Please type some text.")
            continue

        # Speak the text entered by the user.
        print(f"Speaking: {user_text}")
        engine.say(user_text)
        engine.runAndWait()


def main():
    print("=== Text-to-Speech Demo ===")

    # Create one engine instance.
    engine = pyttsx3.init("sapi5")

    # Show available voices.
    list_voices(engine)

    # Configure the engine.
    # Try voice_index=1 or voice_index=2 for English voices on your system.
    configure_engine(
        engine=engine,
        voice_index=1,
        rate=150,
        volume=1.0
    )

    # Start the continuous speech loop directly.
    speak_loop(engine)

    # Stop the engine at the end.
    engine.stop()


if __name__ == "__main__":
    main()