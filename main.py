import whisper
import os
import shutil
import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip
from tqdm import tqdm

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_THICKNESS = 2
BACKGROUND_ALPHA = 0.7
TEXT_PADDING = 20
LINE_SPACING = 10

class VideoTranscriber:
    def __init__(self, model_path, video_path, font_scale=1.5, display_mode='single'):
        self.model = whisper.load_model(model_path)
        self.video_path = video_path
        self.audio_path = ''
        self.text_array = []
        self.fps = 0
        self.cropped_width = 0
        self.frame_height = 0
        self.font_scale = font_scale
        self.display_mode = display_mode
        self.aspect_ratio = 16/9
        self.space_width = None

    def transcribe_video(self):
        print('Début de la transcription')
        try:
            result = self.model.transcribe(self.audio_path)
            cap = cv2.VideoCapture(self.video_path)
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            
            desired_width = int(self.frame_height * self.aspect_ratio)
            self.cropped_width = min(desired_width, original_width)
            
            # Pré-calculer la largeur d'un espace
            self.space_width = cv2.getTextSize(' ', FONT, self.font_scale, FONT_THICKNESS)[0][0]

            for segment in tqdm(result["segments"], desc="Traitement des segments"):
                text = segment["text"].strip()
                start = segment["start"]
                end = segment["end"]
                self.process_segment(text, start, end)
            
            cap.release()
            print('Transcription terminée')
        except Exception as e:
            print(f"Erreur : {str(e)}")

    def process_segment(self, text, start, end):
        total_frames = int((end - start) * self.fps)
        start_frame = int(start * self.fps)
        
        if self.display_mode == 'single':
            words = text.split()
            current_start = start_frame
            total_chars = len(text)
            for word in words:
                word_duration = int((len(word) / total_chars) * total_frames)
                self.text_array.append([word, current_start, current_start + word_duration])
                current_start += word_duration
        else:
            lines = self.split_text_to_lines(text)
            segment_duration = total_frames
            for line in lines:
                self.text_array.append([line, start_frame, start_frame + segment_duration])

    def split_text_to_lines(self, text):
        lines = []
        current_line = []
        current_width = 0
        
        for word in text.split():
            word_width = cv2.getTextSize(word, FONT, self.font_scale, FONT_THICKNESS)[0][0]
            
            if current_line:
                tentative_width = current_width + self.space_width + word_width
            else:
                tentative_width = word_width
            
            if tentative_width <= self.cropped_width * 0.9:  # Marge de 10%
                if current_line:
                    current_line.append(word)
                    current_width += self.space_width + word_width
                else:
                    current_line.append(word)
                    current_width = word_width
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_width = word_width
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines

    def extract_audio(self):
        print('Extraction audio...')
        try:
            self.audio_path = os.path.splitext(self.video_path)[0] + "_audio.mp3"
            video = VideoFileClip(self.video_path)
            video.audio.write_audiofile(self.audio_path)
            print('Audio extrait')
        except Exception as e:
            print(f"Erreur audio : {str(e)}")

    def extract_frames(self, output_folder):
        print('Extraction des images...')
        try:
            cap = cv2.VideoCapture(self.video_path)
            os.makedirs(output_folder, exist_ok=True)
            frame_count = 0
            
            while True:
                success, frame = cap.read()
                if not success:
                    break
                
                original_width = frame.shape[1]
                crop_x = (original_width - self.cropped_width) // 2
                frame = frame[:, crop_x:crop_x + self.cropped_width]
                
                self.add_subtitles_to_frame(frame, frame_count)
                
                cv2.imwrite(os.path.join(output_folder, f"{frame_count:05d}.jpg"), frame)
                frame_count += 1
            
            cap.release()
            print(f"{frame_count} images extraites")
        except Exception as e:
            print(f"Erreur images : {str(e)}")

    def add_subtitles_to_frame(self, frame, frame_number):
        active_subtitles = [text for text, start, end in self.text_array if start <= frame_number <= end]
        
        if not active_subtitles:
            return

        y_position = frame.shape[0] - 50  # Commence 50px du bas
        
        for text in reversed(active_subtitles):
            text_size = cv2.getTextSize(text, FONT, self.font_scale, FONT_THICKNESS)[0]
            text_h = text_size[1]
            
            # Calcul position fond
            bg_x1 = (frame.shape[1] - text_size[0]) // 2 - TEXT_PADDING
            bg_y1 = y_position - text_h - 2 * TEXT_PADDING
            bg_x2 = bg_x1 + text_size[0] + 2 * TEXT_PADDING
            bg_y2 = y_position
            
            # Dessin fond
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0,0,0), -1)
            cv2.addWeighted(overlay, BACKGROUND_ALPHA, frame, 1 - BACKGROUND_ALPHA, 0, frame)
            
            # Dessin texte
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = y_position - TEXT_PADDING
            cv2.putText(frame, text, (text_x, text_y), FONT, self.font_scale, (0,0,0), FONT_THICKNESS+2, cv2.LINE_AA)
            cv2.putText(frame, text, (text_x, text_y), FONT, self.font_scale, (255,255,255), FONT_THICKNESS, cv2.LINE_AA)
            
            # Ajustement position Y pour prochaine ligne
            y_position = bg_y1 - LINE_SPACING

    def create_video(self, output_path):
        print('Création vidéo...')
        try:
            temp_folder = os.path.join(os.path.dirname(self.video_path), "temp_frames")
            self.extract_frames(temp_folder)
            
            images = [img for img in os.listdir(temp_folder) if img.endswith(".jpg")]
            images.sort()
            
            clip = ImageSequenceClip([os.path.join(temp_folder, img) for img in images], fps=self.fps)
            audio = AudioFileClip(self.audio_path)
            clip = clip.set_audio(audio)
            clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
            
            shutil.rmtree(temp_folder)
            if os.path.exists(self.audio_path):
                os.remove(self.audio_path)
            print('Vidéo créée')
        except Exception as e:
            print(f"Erreur vidéo : {str(e)}")

# Exemple d'utilisation
if __name__ == "__main__":
    transcriber = VideoTranscriber(
        model_path="base",
        video_path="0304.mp4",
        font_scale=1.5,
        display_mode='multiple'
    )
    transcriber.extract_audio()
    transcriber.transcribe_video()
    transcriber.create_video("output_video_sync.mp4")