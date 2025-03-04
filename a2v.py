import cv2
from moviepy.editor import AudioFileClip, VideoFileClip


def create_video_with_audio(input_image_path, output_video_path, audio_path):
    # Read the input image
    image = cv2.imread(input_image_path)
    # Load the audio file
    audio_clip = AudioFileClip(audio_path)

    # Get the dimensions of the image
    height, width, layers = image.shape

    # Get the duration of the audio clip
    video_duration = audio_clip.duration
    # Set frames per second for the video
    fps = 1

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Calculate the total number of frames for the video
    total_frames = video_duration * fps
    # Write the image to the video for each frame
    for _ in range(int(total_frames)):
        video_writer.write(image)
    # Release the video writer
    video_writer.release()

    # Clip the audio to match the video duration
    clipped_audio = audio_clip.subclip(0, video_duration)

    # Load the video file
    video_clip = VideoFileClip(output_video_path)
    # Set the audio for the video
    video_with_audio = video_clip.set_audio(clipped_audio)
    # Write the final video file with audio
    video_with_audio.write_videofile(output_video_path, codec="libx264", audio_codec="aac")

    # Optional: Use the following codec format to play video on phone, WhatsApp, etc.
    # video_with_audio.write_videofile(output_video_path, codec='h264', audio_codec='aac')

    # Print a message indicating the video has been saved
    print(f"Video with sound saved as {output_video_path}")

# Define paths for input image, output video, and audio file
input_image_path = "sound.png"
output_video_path = "output_video.mp4"
audio_path = "harvard.wav"

# Call the method to create video with audio
create_video_with_audio(input_image_path, output_video_path, audio_path)
