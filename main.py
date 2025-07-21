import cv2
import pyttsx3
import time
from google.cloud import vision
import os

# Setting up the credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\Arun JH\Desktop\Object_Label_Reader\google_key.json'

class ProductLabelReader:
    # Startup
    def __init__(self):
        # TTS engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Slower speech for blind users
        
        # Initializing Google Vision client
        try:
            self.vision_client = vision.ImageAnnotatorClient()
            self.speak(" Hey Vaibhavi, Product reader initialized. Press SPACE to scan, Q to quit.")
        except Exception as e:
            print(f"Error initializing Google Vision: {e}")
            self.speak("Error: Could not connect")
            exit()
    
    #The real talk
    def speak(self, text):
        """text-to-speech using pyttsx3 only"""
        print(f"Speaking: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
    
    #To See
    def detect_text(self, frame):
        """Extract text from image using Google Vision API heheeee:)"""
        try:
            # Encode frame as JPEG
            success, encoded_image = cv2.imencode('.jpg', frame)
            if not success:
                return None
            
            # Send to Google Vision
            content = encoded_image.tobytes() # turns the image into byte stream
            image = vision.Image(content=content)
            response = self.vision_client.text_detection(image=image)
            
            # Handle API errors
            if response.error.message:
                print(f"API error: {response.error.message}")
                return None
            
            # Extract all text regions with positions
            texts = response.text_annotations  #contains a list of text detections
            if texts:
                # Get all individual text regions (skip first one which is full text)
                text_regions = [] #full text 
                for text in texts[1:]:     
                    region_text = text.description.strip()
                    
                    # Get bounding box info for size filtering
                    vertices = text.bounding_poly.vertices
                    if len(vertices) >= 2: #ensures atleast 2 corners 
                        width = abs(vertices[1].x - vertices[0].x)
                        height = abs(vertices[2].y - vertices[0].y)
                        area = width * height
                        
                        text_regions.append({
                            'text': region_text,
                            'area': area,
                            'width': width,
                            'height': height
                        })
                
                return text_regions
            
            return None
            
        except Exception as e:
            print(f"Text detection error: {e}")
            return None
    
    def filter_product_name(self, text_regions):
        """filtering to find the most likely product name"""
        if not text_regions:
            return None
        
        # Filtering out  the noise
        filtered_regions = []
        for region in text_regions:
            text = region['text'].strip()
            
            # Skipping if too short or just numbers
            if len(text) < 2:
                continue
            if text.isdigit():
                continue
            
            # Skipping common noise words
            noise_words = ['ml', 'mg', 'kg', 'oz', 'lb', 'ingredients', 'nutrition', 
                          'contains', 'warning', 'caution', 'exp', 'best', 'before',
                          'net', 'wt', 'weight', 'serving', 'calories']
            
            if any(noise in text.lower() for noise in noise_words):
                continue
            
            # ignoring small texts
            if region['area'] < 1000:  # Adjust this threshold as needed
                continue
            
            filtered_regions.append(region)
        
        if not filtered_regions:
            return "No product name found"
        
        # Sort by text size( bigger text is usually product name)
        filtered_regions.sort(key=lambda x: x['area'], reverse=True)
        
        # the largest text that looks like a product name
        for region in filtered_regions[:3]:  # Checking top 3 largest
            text = region['text']
            
            # Prefer text with mixed case or all caps(brand names)
            if text.isupper() or any(c.isupper() for c in text):
                return text
        
        # If no mixed case found, return the largest text
        return filtered_regions[0]['text']
    
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
            
            # Display camera 
            cv2.putText(frame, "Press SPACE to scan, Q to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Product Label Reader", frame)
            
            # key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Spacebar to capture
                self.speak("Scanning...")
                
                # Detect text regions
                text_regions = self.detect_text(frame)
                
                if text_regions:
                    # Filter to find product name
                    product_name = self.filter_product_name(text_regions)
                    
                    if product_name and product_name != "No product name found":
                        self.speak(f"Product: {product_name}")
                        
                        # Debug info in console
                        print(f"All detected text regions: {len(text_regions)}")
                        for i, region in enumerate(text_regions[:5]):  # Show first 5 for debugging
                            print(f"  {i+1}: {region['text']} (area: {region['area']})")
                    else:
                        self.speak("Could not identify product name. Try different angle or lighting.")
                else:
                    self.speak("No text detected. Move closer to the product.")
                
                # Small delay to prevent accidental double-scans
                time.sleep(1)
            
            elif key == ord('q'):  # Quit
                break
        
        # Cleanup
        cam.release()
        cv2.destroyAllWindows()
        self.speak("Product reader closed.")

if __name__ == "__main__":
    reader = ProductLabelReader()
    reader.run()