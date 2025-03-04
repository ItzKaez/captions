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
        self.char_width = 0

    def transcribe_video(self):
        print('Début de la transcription')
        try:
            result = self.model.transcribe(self.audio_path)
            cap = cv2.VideoCapture(self.video_path)
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calcul du cadrage
            desired_width = int(self.frame_height * self.aspect_ratio)
            self.cropped_width = min(desired_width, original_width)
            
            # Initialisation de la largeur des caractères
            test_text = result["segments"][0]["text"][:10] or "Hello"
            textsize = cv2.getTextSize(test_text, FONT, self.font_scale, FONT_THICKNESS)[0]
            self.char_width = textsize[0] / len(test_text)

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
        total_chars = len(text)
        
        if self.display_mode == 'single':
            words = text.split()
            current_start = start_frame
            for word in words:
                word_length = len(word)
                word_duration = int((word_length / total_chars) * total_frames)
                self.text_array.append([
                    word,
                    current_start,
                    current_start + word_duration
                ])
                current_start += word_duration
        else:
            lines = []
            current_line = []
            current_width = 0
            
            for word in text.split():
                word_width = len(word) * self.char_width
                space_width = self.char_width  # Espace approximatif
                
                if current_width + word_width + space_width < self.cropped_width * 0.9:
                    current_line.append(word)
                    current_width += word_width + space_width
                else:
                    lines.append((' '.join(current_line), start_frame))
                    current_line = [word]
                    current_width = word_width
            
            if current_line:
                lines.append((' '.join(current_line), start_frame))
            
            # Répartition temporelle précise
            for line, line_start in lines:
                line_length = len(line)
                line_duration = int((line_length / total_chars) * total_frames)
                self.text_array.append([
                    line,
                    line_start,
                    line_start + line_duration
                ])

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
                
                # Cadrage
                original_width = frame.shape[1]
                crop_x = (original_width - self.cropped_width) // 2
                frame = frame[:, crop_x:crop_x + self.cropped_width]
                
                # Ajout des sous-titres
                self.add_subtitles_to_frame(frame, frame_count)
                
                cv2.imwrite(os.path.join(output_folder, f"{frame_count:05d}.jpg"), frame)
                frame_count += 1
            
            cap.release()
            print(f"{frame_count} images extraites")
        except Exception as e:
            print(f"Erreur images : {str(e)}")

    def add_subtitles_to_frame(self, frame, frame_number):
        for text, start, end in self.text_array:
            if start <= frame_number <= end:
                position = 'center'
                self.draw_text_with_background(frame, text, position)
                break

    def draw_text_with_background(self, frame, text, position):
        text_size = cv2.getTextSize(text, FONT, self.font_scale, FONT_THICKNESS)[0]
        text_w, text_h = text_size
        
        # Positionnement dynamique
        if position == 'center':
            pos_y = int(frame.shape[0]/2 + text_h/2)
        else:
            pos_y = frame.shape[0] - 50  # 50 pixels du bas
        
        pos_x = int((frame.shape[1] - text_w)/2)
        
        # Fond semi-transparent
        bg_x1 = pos_x - TEXT_PADDING
        bg_y1 = pos_y - text_h - TEXT_PADDING
        bg_x2 = pos_x + text_w + TEXT_PADDING
        bg_y2 = pos_y + TEXT_PADDING
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, BACKGROUND_ALPHA, frame, 1 - BACKGROUND_ALPHA, 0, frame)
        
        # Texte avec contour net
        cv2.putText(frame, text, (pos_x, pos_y), 
                    FONT, self.font_scale, (0,0,0), FONT_THICKNESS + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (pos_x, pos_y), 
                    FONT, self.font_scale, (255,255,255), FONT_THICKNESS, cv2.LINE_AA)

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
        video_path="subtitle_generator/0304.mp4",
        font_scale=1.5,
        display_mode='single'  # Testez avec 'single' ou 'multiple'
    )
    transcriber.extract_audio()
    transcriber.transcribe_video()
    transcriber.create_video("output_video_sync.mp4")