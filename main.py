import whisper
import os
import shutil
import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip
from tqdm import tqdm
from difflib import SequenceMatcher
from PIL import ImageFont, ImageDraw, Image

# Font configuration
FONT_SIZE = 45  # Increased font size for better visibility
print("Trying to load font...")

# List of possible font paths
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    "arial.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]

# Try to load a suitable font
FONT = None
for font_path in FONT_PATHS:
    try:
        FONT = ImageFont.truetype(font_path, FONT_SIZE)
        print(f"Successfully loaded font: {font_path}")
        break
    except Exception as e:
        print(f"Failed to load font {font_path}: {str(e)}")

# If no font could be loaded, use default
if FONT is None:
    print("Using default PIL font as fallback")
    FONT = ImageFont.load_default()
BACKGROUND_ALPHA = 0.7
TEXT_PADDING = 30  # Increased padding for better readability
LINE_SPACING = 15  # Increased line spacing
TEXT_COLOR = (255, 255, 255, 255)  # White with full opacity
OUTLINE_COLOR = (0, 0, 0, 255)     # Black with full opacity

class VideoTranscriber:
    def __init__(self, model_path, video_path, display_mode='single', custom_transcription=None, 
                 similarity_threshold=0.7, debug=False):
        self.model = whisper.load_model(model_path)
        self.video_path = video_path
        self.audio_path = ''
        self.text_array = []
        self.fps = 0
        self.cropped_width = 0
        self.frame_height = 0
        self.display_mode = display_mode
        self.aspect_ratio = 16/9
        self.custom_transcription = custom_transcription
        self.custom_words = []
        self.custom_word_index = 0
        self.debug = debug
        self.similarity_threshold = similarity_threshold
        
        # If custom transcription is provided, preprocess it
        if self.custom_transcription:
            # Split into words and clean up
            self.custom_words = [word.strip() for word in self.custom_transcription.split() if word.strip()]
            if self.debug:
                print(f"Loaded {len(self.custom_words)} words from custom transcription")
                print("First few words:", self.custom_words[:5])

    def get_word_similarity(self, word1, word2):
        """
        Calculate similarity ratio between two words using SequenceMatcher.
        """
        return SequenceMatcher(None, word1.lower(), word2.lower()).ratio()

    def find_best_matching_word(self, whisper_word, segment_position, total_segments):
        """
        Find the best matching word from custom transcription considering position context.
        
        Args:
            whisper_word: The word detected by Whisper
            segment_position: Current segment index
            total_segments: Total number of segments
        """
        if not self.custom_words or self.custom_word_index >= len(self.custom_words):
            return whisper_word, 0

        # Calculate the expected position ratio (0 to 1) in the text
        expected_position_ratio = segment_position / total_segments
        expected_index = int(expected_position_ratio * len(self.custom_words))
        
        # Define a wider search window around the expected position
        window_size = len(self.custom_words) // 4  # 25% of text length
        start_idx = max(0, expected_index - window_size)
        end_idx = min(len(self.custom_words), expected_index + window_size)
        
        if self.debug:
            print(f"\nSearching for '{whisper_word}' around position {expected_index} (window: {start_idx} to {end_idx})")
        
        best_match = whisper_word
        best_score = 0
        best_idx = self.custom_word_index

        # Search for the best matching word within the window
        for idx in range(start_idx, end_idx):
            custom_word = self.custom_words[idx]
            base_similarity = self.get_word_similarity(whisper_word, custom_word)
            
            # Apply position-based penalty
            position_diff_ratio = abs(idx - expected_index) / window_size
            position_penalty = 1 - (position_diff_ratio * 0.5)  # Max 50% penalty for position
            
            # Final score combines similarity and position
            adjusted_score = base_similarity * position_penalty
            
            if self.debug and adjusted_score > 0.5:
                print(f"  Comparing with '{custom_word}' at idx {idx}:")
                print(f"    Base similarity: {base_similarity:.2f}")
                print(f"    Position penalty: {position_penalty:.2f}")
                print(f"    Final score: {adjusted_score:.2f}")
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_match = custom_word
                best_idx = idx

        # Update the index if we found a good match
        if best_score >= self.similarity_threshold:
            self.custom_word_index = best_idx + 1
            if self.debug:
                print(f"  Selected '{best_match}' (score: {best_score:.2f})")
            return best_match, best_score
        
        if self.debug:
            print(f"  No good match found, keeping '{whisper_word}'")
        return whisper_word, best_score

    def refine_segment_text(self, whisper_text, segment_index, total_segments):
        """
        Refine the segment text using fuzzy matching with custom transcription.
        Returns the refined text and updates the custom word index.
        """
        if not self.custom_transcription or not self.custom_words:
            return whisper_text

        if self.debug:
            print(f"\nRefining segment {segment_index + 1}/{total_segments}: '{whisper_text}'")

        whisper_words = whisper_text.split()
        refined_words = []
        
        for whisper_word in whisper_words:
            # Find the best matching word from custom transcription
            best_match, score = self.find_best_matching_word(whisper_word, segment_index, total_segments)
            refined_words.append(best_match)
            
            if score >= self.similarity_threshold:
                print(f"Replaced '{whisper_word}' with '{best_match}' (score: {score:.2f})")
            
        return ' '.join(refined_words)

    def transcribe_video(self):
        print('Début de la transcription')
        try:
            # Use word-level timestamps for better synchronization
            result = self.model.transcribe(self.audio_path, word_timestamps=True)
            cap = cv2.VideoCapture(self.video_path)
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            
            desired_width = int(self.frame_height * self.aspect_ratio)
            self.cropped_width = min(desired_width, original_width)
            
            segments = result["segments"]
            total_segments = len(segments)
            
            for i, segment in enumerate(tqdm(segments, desc="Traitement des segments")):
                # Get word-level timing information
                words_with_timing = segment.get("words", [])
                if not words_with_timing:
                    # Fallback if word timing is not available
                    whisper_text = segment["text"].strip()
                    refined_text = self.refine_segment_text(whisper_text, i, total_segments)
                    start = segment["start"]
                    end = segment["end"]
                    self.process_segment(refined_text, start, end)
                else:
                    # Process words with their timing information
                    self.process_segment_with_word_timing(words_with_timing, i, total_segments)
            
            if self.custom_transcription and self.custom_word_index < len(self.custom_words):
                print(f"Warning: {len(self.custom_words) - self.custom_word_index} words from custom transcription were not used")
            
            cap.release()
            print('Transcription terminée')
        except Exception as e:
            print(f"Erreur : {str(e)}")

    def process_segment_with_word_timing(self, words_with_timing, segment_index, total_segments):
        """Process a segment using word-level timing information."""
        if not words_with_timing:
            return

        try:
            # Create a temporary image for text measurements
            temp_img = Image.new('RGBA', (self.cropped_width, 100), (255, 255, 255, 0))
            draw = ImageDraw.Draw(temp_img)
            
            current_line = []
            current_line_words = []
            lines = []
            line_timings = []
            
            for word_info in words_with_timing:
                word = word_info["word"].strip()
                if not word:
                    continue
                
                # Try to add word to current line
                test_line = ' '.join(current_line + [word])
                bbox = draw.textbbox((0, 0), test_line, font=FONT)
                text_width = bbox[2] - bbox[0]
                
                if text_width <= self.cropped_width * 0.9:  # 10% margin
                    current_line.append(word)
                    current_line_words.append(word_info)
                else:
                    if current_line:
                        # Add completed line with its timing
                        line_text = ' '.join(current_line)
                        start_time = current_line_words[0]["start"]
                        end_time = current_line_words[-1]["end"]
                        lines.append(line_text)
                        line_timings.append((start_time, end_time))
                    
                    # Start new line with current word
                    current_line = [word]
                    current_line_words = [word_info]
            
            # Add the last line if there are remaining words
            if current_line:
                line_text = ' '.join(current_line)
                start_time = current_line_words[0]["start"]
                end_time = current_line_words[-1]["end"]
                lines.append(line_text)
                line_timings.append((start_time, end_time))
            
            # Add each line as a separate segment with its timing
            for line, (start_time, end_time) in zip(lines, line_timings):
                start_frame = int(start_time * self.fps)
                end_frame = int(end_time * self.fps)
                
                if self.text_array:
                    # Ensure no overlap with previous segment
                    last_end = self.text_array[-1][2]
                    if start_frame < last_end:
                        start_frame = last_end + 1
                
                if start_frame < end_frame:  # Ensure valid duration
                    self.text_array.append([line, start_frame, end_frame])
                    if self.debug:
                        print(f"Added line: '{line}' (frames {start_frame} to {end_frame})")
        
        except Exception as e:
            print(f"Error processing segment with word timing: {str(e)}")

    def process_segment(self, text, start, end):
        """Fallback process_segment method when word timing is not available."""
        if self.debug:
            print(f"Processing segment without word timing: '{text}' ({start:.2f}s to {end:.2f}s)")
            
        total_frames = int((end - start) * self.fps)
        start_frame = int(start * self.fps)
        
        try:
            # Create a temporary image for text measurements
            temp_img = Image.new('RGBA', (self.cropped_width, 100), (255, 255, 255, 0))
            draw = ImageDraw.Draw(temp_img)
            
            words = text.split()
            current_line = []
            lines = []
            
            # Split text into lines that fit the screen width
            for word in words:
                test_line = ' '.join(current_line + [word])
                bbox = draw.textbbox((0, 0), test_line, font=FONT)
                text_width = bbox[2] - bbox[0]
                
                if text_width <= self.cropped_width * 0.9:  # 10% margin
                    current_line.append(word)
                else:
                    if current_line:  # Add current line if not empty
                        lines.append(' '.join(current_line))
                    current_line = [word]  # Start new line with current word
            
            # Add the last line if there are remaining words
            if current_line:
                lines.append(' '.join(current_line))
            
            if not lines:  # If no lines were created (empty text)
                return
                
            # Calculate frames per line to maintain synchronization
            frames_per_line = total_frames // len(lines)
            remaining_frames = total_frames % len(lines)
            
            # Add each line as a separate segment with appropriate timing
            current_start = start_frame
            for i, line in enumerate(lines):
                # Add extra frame to earlier segments if we have remaining frames
                extra_frames = 1 if i < remaining_frames else 0
                segment_frames = frames_per_line + extra_frames
                
                if self.text_array:
                    # Ensure no overlap with previous segment
                    last_end = self.text_array[-1][2]
                    if current_start < last_end:
                        current_start = last_end + 1
                
                # Only add the segment if we have valid duration
                if current_start + segment_frames > current_start:
                    self.text_array.append([line, current_start, current_start + segment_frames])
                    if self.debug:
                        print(f"Added line: '{line}' (frames {current_start} to {current_start + segment_frames})")
                    current_start += segment_frames
            
        except Exception as e:
            print(f"Error processing segment: {str(e)}")

    def add_subtitles_to_frame(self, frame, frame_number):
        # Get all active subtitle entries with their timing information
        active_entries = [[text, start, end] for text, start, end in self.text_array if start <= frame_number <= end]
        
        if not active_entries:
            return

        # Select only the most recent subtitle (highest start frame)
        active_text = max(active_entries, key=lambda entry: entry[1])[0]

        if self.debug:
            print(f"Processing frame {frame_number} with subtitle: {active_text}")
        
        SUBTITLE_MARGIN_BOTTOM = 750  # Distance from bottom of frame

        try:
            # Convert the frame from BGR to RGBA for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb).convert('RGBA')
            draw = ImageDraw.Draw(pil_image)  # Need this for text measurements
            
            if self.debug:
                print(f"Frame conversion successful - Shape: {frame.shape}, Mode: {pil_image.mode}")

            # Get text dimensions
            bbox = draw.textbbox((0, 0), active_text, font=FONT)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Calculate text position (centered horizontally, fixed distance from bottom)
            text_x = (frame.shape[1] - text_width) // 2
            text_y = frame.shape[0] - (text_height + TEXT_PADDING + SUBTITLE_MARGIN_BOTTOM)

            # Calculate background position
            bg_x1 = (frame.shape[1] - text_width) // 2 - TEXT_PADDING
            bg_y1 = text_y - TEXT_PADDING
            bg_x2 = bg_x1 + text_width + 2 * TEXT_PADDING
            bg_y2 = text_y + text_height + TEXT_PADDING
            
            # Draw semi-transparent black background
            overlay = Image.new('RGBA', (frame.shape[1], frame.shape[0]), (0, 0, 0, 0))
            d = ImageDraw.Draw(overlay)
            d.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], 
                      fill=(0, 0, 0, int(255 * BACKGROUND_ALPHA)))

            if self.debug:
                print(f"Drawing text: '{active_text}' at position ({text_x}, {text_y})")
                print(f"Text dimensions: width={text_width}, height={text_height}")
                print(f"Background rectangle: ({bg_x1}, {bg_y1}) to ({bg_x2}, {bg_y2})")

            # Draw text with outline
            text_overlay = Image.new('RGBA', (frame.shape[1], frame.shape[0]), (0, 0, 0, 0))
            text_draw = ImageDraw.Draw(text_overlay)

            # Draw black outline
            outline_size = 3  # Increased outline size for better visibility
            for dx in range(-outline_size, outline_size + 1):
                for dy in range(-outline_size, outline_size + 1):
                    if dx != 0 or dy != 0:
                        text_draw.text((text_x + dx, text_y + dy), active_text, 
                                     font=FONT, fill=OUTLINE_COLOR)

            # Draw white text
            text_draw.text((text_x, text_y), active_text, font=FONT, fill=TEXT_COLOR)

            # Composite background and text onto the frame
            pil_image = Image.alpha_composite(pil_image, overlay)
            pil_image = Image.alpha_composite(pil_image, text_overlay)

            # Convert back to BGR for OpenCV (removing alpha channel)
            frame_with_text = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
            
            # Copy the result back to the original frame
            frame[:] = frame_with_text
            
            if self.debug:
                print(f"Frame {frame_number}: Successfully rendered text and converted back to BGR")
                print(f"Successfully added subtitles to frame {frame_number}")

        except Exception as e:
            print(f"Error adding subtitles to frame: {str(e)}")

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

# Example usage
if __name__ == "__main__":
    # Example with custom transcription for a music video
    custom_text = """
un j dans la main gauche elle attrape ma main droite pour que je la pénètre
Merco dernier cri, j'ai pas besoin d'la conduire mais j'fais quand même belek
Mais j'fais quand même belek
Le shit, les xans par la fenêtre 
        """
    
    print("Starting video transcription with improved subtitle rendering...")
    
    transcriber = VideoTranscriber(
        model_path="base",
        video_path="0304.mp4",
        display_mode='multiple',
        custom_transcription=custom_text,
        similarity_threshold=0.7,  # Adjust this threshold to control matching sensitivity
        debug=False  # Enable debug mode for detailed logging
    )
    
    print("\nExtracting audio...")
    transcriber.extract_audio()
    
    print("\nTranscribing video...")
    transcriber.transcribe_video()
    
    print("\nCreating final video with subtitles...")
    transcriber.create_video("output_video_sync.mp4")
    
    print("\nProcessing complete! The output video has been saved as 'output_video_sync.mp4'")
