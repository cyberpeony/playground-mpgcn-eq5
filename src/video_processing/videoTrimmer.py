import os 
import pandas as pd 
from moviepy.editor import VideoFileClip


def time_to_seconds(t: str) -> float:
    """
    Docstring for time_to_seconds
    
    :param time_str: Description
    """
    parts = t.split(':')
    hours, minutes, seconds = parts
    return float(hours) * 3600 + float(minutes) * 60 + float(seconds)

def trim_video(video_directory:str, scene_id: str, video_id: str, t_start: str, t_end: str, label: str) -> str:
    """
    Docstring for trim_video

    """

    output_path = "/home/sformador/equipo5/Socio-Formador-IA-Avanzada/video_processing/trimmedClips"

    os.makedirs(output_path, exist_ok=True)

    video_path = os.path.join(video_directory, video_id)

    if not os.path.exists(video_path):
        print(f"Video file {video_path} does not exist.")
        return ""
    
    print(f"Processing video: {video_path}")


    scene_name = f"{scene_id}_{label}"

    trimmed_output_path = os.path.join(output_path, f"{scene_name}.mp4")

    if os.path.exists(trimmed_output_path):
        print(f"Trimmed video already exists: {trimmed_output_path}")
        return scene_name
    
    start_sec = time_to_seconds(t_start)
    end_sec = time_to_seconds(t_end)

    try:
        video = VideoFileClip(video_path).subclip(start_sec, end_sec)
        video.write_videofile(trimmed_output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        print(f"Trimmed video saved to: {trimmed_output_path}")
        return scene_name
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return ""
