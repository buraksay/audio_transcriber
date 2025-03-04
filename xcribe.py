import argparse
import logging as log
import math
import os
import sys

import cv2
import ffmpeg
from faster_whisper import WhisperModel
from moviepy.editor import AudioFileClip, VideoFileClip

# Configure logging to show INFO level messages
log.basicConfig(level=log.INFO)

def transcribe(audio_file, model):
    """
    Transcribe an audio file using the Whisper model.
    
    Args:
        audio_file (str): Path to the audio file to transcribe
        model (WhisperModel): Pre-loaded Whisper model instance
        
    Returns:
        tuple: (language_code, segments) where language_code is the detected language
              and segments contain the transcribed text with timestamps
    """
    segments, info = model.transcribe(audio_file)
    language = info.language
    print("Transcription language", info.language)
    segments = list(segments)
    for segment in segments:
        # Log each transcribed segment with its start and end timestamps
        log.info("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    return language, segments


def format_time(seconds):
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted timestamp in SRT format (HH:MM:SS,mmm)
    """
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:01d},{milliseconds:03d}"

    return formatted_time


def generate_subtitle_file(segments, subtitle_file):
    """
    Generate an SRT subtitle file from transcription segments.
    
    Args:
        segments (list): List of transcription segments with timestamps and text
        subtitle_file (str): Path to the output subtitle file
        
    Returns:
        str: Path to the created subtitle file
    """
    text = ""
    for index, segment in enumerate(segments):
        segment_start = format_time(segment.start)
        segment_end = format_time(segment.end)
        # Format as SRT with index, timestamp range, and text
        text += f"{str(index+1)} \n"
        text += f"{segment_start} --> {segment_end} \n"
        text += f"{segment.text} \n"
        text += "\n"

    f = open(subtitle_file, "w")
    f.write(text)
    f.close()
    log.info(f"Subtitle file saved as {subtitle_file}")
    return subtitle_file


def add_subtitle_to_video(soft_subtitle, subtitle_file, subtitle_language, input_video, output_video):
    """
    Add subtitles to a video using ffmpeg.
    
    Args:
        soft_subtitle (bool): If True, adds subtitles as a separate track that can be toggled on/off.
                             If False, burns subtitles directly into the video.
        subtitle_file (str): Path to the SRT subtitle file
        subtitle_language (str): Language code for the subtitle track metadata
        input_video (str): Path to the input video file
        output_video (str): Path to the output video file with subtitles
    
    Returns:
        str: Path to the output video file
    """
    video_input_stream = ffmpeg.input(input_video)
    subtitle_input_stream = ffmpeg.input(subtitle_file)
    subtitle_track_title = subtitle_file.replace(".srt", "")

    if soft_subtitle:
        # Add subtitles as a separate selectable track without modifying the video
        stream = ffmpeg.output(
            video_input_stream,
            subtitle_input_stream,
            output_video,
            **{"c": "copy", "c:s": "mov_text"},
            **{
                "metadata:s:s:0": f"language={subtitle_language}",
                "metadata:s:s:0": f"title={subtitle_track_title}",
            },
        )
        ffmpeg.run(stream, overwrite_output=True)
    else:
        # Burn subtitles directly into the video frames
        stream = ffmpeg.output(
            video_input_stream, output_video, vf=f"subtitles={subtitle_file}"
        )
        ffmpeg.run(stream, overwrite_output=True)
        
    log.info(f"Video with subtitles saved as {output_video}")
    return output_video


def create_video_with_audio(input_image_file, output_video_file, audio_file):
    """
    Create a video from a static image with audio.
    
    Args:
        input_image_file (str): Path to the image file to use for video
        output_video_file (str): Path to save the output video
        audio_file (str): Path to the audio file to add to the video
    """
    # Read the input image
    image = cv2.imread(input_image_file)

    # Get the dimensions of the image
    height, width, layers = image.shape
    log.info(f"Loaded image with dimensions: {width}x{height}")

    # Load the audio file to determine duration
    audio_clip = AudioFileClip(audio_file)
    video_duration = audio_clip.duration
    # Use a low frame rate since it's a static image
    fps = 1
    log.info(f"Creating video with duration: {video_duration}s")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    # Calculate the total number of frames needed for the video duration
    total_frames = video_duration * fps
    # Write the same image to the video for each frame
    for _ in range(int(total_frames)):
        video_writer.write(image)
    # Release the video writer to finalize the video file
    video_writer.release()

    # Clip the audio to match the video duration (if needed)
    clipped_audio = audio_clip.subclip(0, video_duration)

    # Load the video file to add audio
    video_clip = VideoFileClip(output_video_file)
    # Set the audio for the video
    video_with_audio = video_clip.set_audio(clipped_audio)
    # Write the final video file with audio
    video_with_audio.write_videofile(
        output_video_file, codec="libx264", audio_codec="aac"
    )

    # Note: Alternative codec for better compatibility with mobile devices
    # video_with_audio.write_videofile(output_video_file, codec='h264', audio_codec='aac')

    log.info(f"Video with sound saved as {output_video_file}")


# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Create video with audio and subtitles.")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--audio", type=str, help="Path to the audio file")
group.add_argument("-a", help="Path to the audio file")

args = parser.parse_args()
if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(1)


# Process command line arguments
audio_file = args.audio if args.audio is not None else args.a

# Define paths for input image, output video, and audio file
input_image_file = "voice.jpeg"  # Static image to be used for the video
audio_root, _ = os.path.splitext(audio_file)
output_video_file = f"{audio_root}_video.mp4"  # Video without subtitles
output_video_with_subs_file = f"{audio_root}_video_subbed.mp4"  # Final video with subtitles

# Create a video from the static image and audio file
create_video_with_audio(input_image_file, output_video_file, audio_file)

# Initialize the Whisper model and transcribe the audio
model = WhisperModel("small")
language, segments = transcribe(audio_file, model)
log.info(f"Transcription completed for {audio_file}, detected language: {language}")

# Define subtitle file path using detected language
subtitle_file = f"sub-{audio_root}.{language}.srt"

# Generate the subtitle file from transcription segments
subtitle_file = generate_subtitle_file(segments, subtitle_file)

# Add subtitles to the video (hardcoded/burned-in subtitles)
add_subtitle_to_video(
    soft_subtitle=False,  # False = burn subtitles into video, True = add as selectable track
    subtitle_file=subtitle_file, 
    subtitle_language=language, 
    input_video=output_video_file, 
    output_video=output_video_with_subs_file
)
