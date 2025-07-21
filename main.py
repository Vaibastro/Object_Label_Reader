import cv2
import pyttsx3
import time
from google.cloud import vision
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r  'C:\Users\Arun JH\Desktop\Object_Label_Reader\google_key.json'  

class ProductLabelReader:
    def __init__(self):
        # Initialize TTS engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Slower speech for accessibility
        
        # Initialize Google Vision client
        try:
            self.vision_client = vision.ImageAnnotatorClient()
            self.speak("Product reader initialized. Press SPACE to scan, Q to quit.")
        except Exception as e:
            print(f"Error initializing Google Vision: {e}")
            self.speak("Error: Could not connect to text detection service")
            exit()
    
    def speak(self, text):
        """Reliable text-to-speech using pyttsx3 only"""
        print(f"Speaking: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
    
    def detect_text(self, frame):
        """Extract text from image using Google Vision API"""
        try:
            # Encode frame as JPEG
            success, encoded_image = cv2.imencode('.jpg', frame)
            if not success:
                return None
            
            # Send to Google Vision
            content = encoded_image.tobytes()
            image = vision.Image(content=content)
            response = self.vision_client.text_detection(image=image)
            
            # Handle API errors
            if response.error.message:
                print(f"Vision API error: {response.error.message}")
                return None
            
            # Extract text
            texts = response.text_annotations
            if texts:
                full_text = texts[0].description.strip()
                return full_text
            
            return None
            
        except Exception as e:
            print(f"Text detection error: {e}")
            return None
    
    def run(self):
        """Main camera loop"""
        # Open camera
        cam = cv2.VideoCapture(0)
        
        if not cam.isOpened():
            self.speak("Error: Could not open camera")
            return
        
        self.speak("Camera ready. Hold product steady and press SPACE to scan.")
        
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            
            # Display camera feed
            cv2.putText(frame, "Press SPACE to scan, Q to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Product Label Reader", frame)
            
         
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  #Spacebar to capture
                self.speak("Scanning...")
                
                # Detect text
                detected_text = self.detect_text(frame)
                
                if detected_text and len(detected_text) > 3:
                    # Cleaning up text 
                    cleaned_text = ' '.join(detected_text.split())
                    self.speak(f"Found: {cleaned_text}")
                else:
                    self.speak("No text found. Try moving closer or adjusting angle.")
                
                #Small delay to prevent accidental 2nd-scans
                time.sleep(1)
            
            elif key == ord('q'):  # Quit
                break
        
        # Cleaning up
        cam.release()
        cv2.destroyAllWindows()
        self.speak("Product reader closed.")

if __name__ == "__main__":
    reader = ProductLabelReader()
    reader.run()