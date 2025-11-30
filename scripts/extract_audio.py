"""
Extract audio from video files in the MELD dataset.

This script extracts audio tracks from .mp4 video files and saves them as .wav files
for audio processing in the emotion recognition pipeline.
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    print("Error: moviepy is not installed.")
    print("Please install it using: pip install moviepy")
    sys.exit(1)


def extract_audio_from_video(video_path, audio_path, sample_rate=16000):
    """
    Extract audio from a video file and save as WAV.
    
    Args:
        video_path (str): Path to input video file
        audio_path (str): Path to output audio file
        sample_rate (int): Audio sample rate in Hz (default: 16000)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        video = VideoFileClip(str(video_path))
        audio = video.audio
        
        if audio is None:
            print(f"Warning: No audio track found in {video_path.name}")
            video.close()
            return False
        
        audio.write_audiofile(
            str(audio_path),
            fps=sample_rate,
            nbytes=2,
            codec='pcm_s16le',
            logger=None  # Suppress moviepy's verbose output
        )
        
        video.close()
        return True
        
    except Exception as e:
        print(f"Error processing {video_path.name}: {str(e)}")
        return False


def main():
    """Main function to extract audio from all videos in the dataset."""
    
    # Define paths
    video_dir = project_root / "data" / "meld" / "video"
    audio_dir = project_root / "data" / "meld" / "audio"
    
    # Create audio directory if it doesn't exist
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if video directory exists
    if not video_dir.exists():
        print(f"Error: Video directory not found at {video_dir}")
        print("Please ensure the MELD dataset videos are placed in data/meld/video/")
        sys.exit(1)
    
    # Get all video files
    video_files = list(video_dir.glob("*.mp4"))
    
    if not video_files:
        print(f"No .mp4 files found in {video_dir}")
        print("Please add video files to the directory.")
        sys.exit(1)
    
    print(f"Found {len(video_files)} video files")
    print(f"Extracting audio to: {audio_dir}")
    print("-" * 60)
    
    # Process each video file
    successful = 0
    failed = 0
    
    for video_path in tqdm(video_files, desc="Extracting audio"):
        # Create corresponding audio filename
        audio_filename = video_path.stem + ".wav"
        audio_path = audio_dir / audio_filename
        
        # Skip if audio file already exists
        if audio_path.exists():
            tqdm.write(f"Skipping {video_path.name} (audio already exists)")
            successful += 1
            continue
        
        # Extract audio
        if extract_audio_from_video(video_path, audio_path):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print("-" * 60)
    print(f"\nExtraction complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(video_files)}")
    print(f"\nAudio files saved to: {audio_dir}")


if __name__ == "__main__":
    main()
