import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
from moviepy import VideoFileClip, concatenate_videoclips
from scenedetect import detect, ContentDetector, AdaptiveDetector
import whisper
from transformers import pipeline
import numpy as np


class VideoContentExtractor:
    def _init_(self, input_video_path):
        self.input_video_path = input_video_path
        self.video = VideoFileClip(input_video_path)
        self.whisper_model = whisper.load_model("base")
        self.analyzer = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            return_all_scores=True
        )
        
    def detect_scenes(self):
        """Detect scene changes in the video"""
        scenes = detect(self.input_video_path, AdaptiveDetector())
        return [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scenes]
    
    def extract_audio_segment(self, start_time, end_time):
        """Extract and transcribe audio for a specific segment"""
        segment = self.video.subclip(start_time, end_time)
        temp_audio_path = "temp_audio.wav"
        segment.audio.write_audiofile(temp_audio_path)
        
        # Transcribe audio using Whisper
        result = self.whisper_model.transcribe(temp_audio_path)
        os.remove(temp_audio_path)
        return result["text"]
    
    def analyze_content_relevance(self, text):
        """Analyze content relevance using BART model"""
        labels = ["engaging", "informative", "action", "highlight"]
        results = self.analyzer(text, candidate_labels=labels)
        # print("Results: ", results)
        # return max(results['scores'])
        return results['scores'][1]
    
    def extract_highlights(self, target_duration=90):
        """Extract meaningful highlights totaling target_duration"""
        scenes = self.detect_scenes()
        scene_scores = []
        
        # Analyze each scene
        for start, end in scenes:
            if end - start < 5:  # Skip very short scenes
                continue
                
            transcript = self.extract_audio_segment(start, end)
            print(" Transcript: ", transcript)
            print("Start: ", start, "End: ", end)
            relevance_score = self.analyze_content_relevance(transcript)
            scene_scores.append({
                'start': start,
                'end': end,
                'score': relevance_score,
                'duration': end - start
            })
        
        # Sort scenes by relevance score
        scene_scores.sort(key=lambda x: x['score'], reverse=True)
        print("Scene Scores: ", scene_scores)
        # Select best scenes that fit within target duration
        selected_scenes = []
        current_duration = 0
        
        for scene in scene_scores:
            if current_duration + scene['duration'] <= target_duration:
                selected_scenes.append(scene)
                current_duration += scene['duration']
            
            if current_duration >= target_duration:
                break
        
        # Sort selected scenes by timestamp
        selected_scenes.sort(key=lambda x: x['start'])

        print("Selected Scenes: ", selected_scenes)
        
        # Create final video
        clips = [self.video.subclip(scene['start'], scene['end']) 
                for scene in selected_scenes]
        final_video = concatenate_videoclips(clips)
        
        return final_video
    
    def process_video(self, output_path):
        """Process the video and save the highlight reel"""
        highlights = self.extract_highlights()
        highlights.write_videofile(output_path, codec='libx264')
        self.video.close()

# Usage example
if __name__ == "_main_":
    extractor = VideoContentExtractor("input_video_1.mp4")
    extractor.process_video("output_highlights_1.mp4")