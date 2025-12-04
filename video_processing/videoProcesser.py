from videoTrimmer import trim_video
from skeletonExtractor import skeletonExtractor
import pandas as pd

VIDEO_DIRECTORY = "/home/sformador/equipo5/Socio-Formador-IA-Avanzada/Downloads"
TRIMMED_CLIPS_DIRECTORY = "/home/sformador/equipo5/Socio-Formador-IA-Avanzada/video_processing/trimmedClips"
TRAINING_CSV_DIRECTORY = "/home/sformador/equipo5/Socio-Formador-IA-Avanzada/video_processing/balanced.csv"



def main():
    print("Starting video processing pipeline...")

    df = pd.read_csv(TRAINING_CSV_DIRECTORY)

    for idx, row in df.iterrows():
        scene_id = row['scene_id']
        video_id = row['video_id']
        t_start = row['t_start']
        t_end = row['t_end']
        label = row['label']

        scene_name = trim_video(VIDEO_DIRECTORY, scene_id, video_id, t_start, t_end, label)

        if scene_name:
            skeletonExtractor(TRIMMED_CLIPS_DIRECTORY, scene_name)
        else:
            print(f"Video trimming failed for {video_id}; skipping skeleton extraction.")

    

if __name__ == "__main__":
    main()