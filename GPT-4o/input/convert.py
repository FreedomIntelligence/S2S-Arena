from pydub import AudioSegment
import os


def convert_audio_bitrate(input_file, output_file, bitrate="256k"):
    # Load the audio file
    try:
        audio = AudioSegment.from_file(input_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Export the audio with specified bitrate
    try:
        audio.export(output_file, format="wav", bitrate=bitrate)
        print(f"Audio file saved as {output_file} with bitrate {bitrate}")
    except Exception as e:
        print(f"Error exporting file: {e}")


if __name__ == "__main__":
    input_path = "english1.wav"
    output_path = "english1_c.wav"

    # Ensure output has proper extension
    if not output_path.lower().endswith(".wav"):
        print("Error: Output file must have .wav extension.")
    elif not os.path.isfile(input_path):
        print("Error: Input file does not exist.")
    else:
        convert_audio_bitrate(input_path, output_path, bitrate="256k")