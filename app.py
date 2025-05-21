import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, Response, render_template, jsonify
import threading
import time
import json

app = Flask(__name__)

# Global variables
keyboard_layout = [
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/'],
    ['SPACE', 'BACKSPACE']
]

# Global state
detected_keys = []
current_text = ""
lock = threading.Lock()

# MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize video capture
cap = None
last_key_press_time = 0
key_cooldown = 0.8  # seconds

# Create a folder for templates if it doesn't exist
import os
if not os.path.exists('templates'):
    os.makedirs('templates')

# Create a folder for static files if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

# Write the HTML template
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Air Keyboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
        .description {
            margin-bottom: 20px;
            color: #666;
            text-align: center;
        }
        .video-text-container {
            display: flex;
            flex-direction: column;
            width: 100%;
            align-items: center;
        }
        #videoElement {
            width: 100%;
            max-width: 800px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        .text-container {
            width: 100%;
            max-width: 800px;
            margin-top: 20px;
        }
        #textOutput {
            width: 100%;
            min-height: 100px;
            padding: 10px;
            font-size: 18px;
            border: 1px solid #ccc;
            border-radius: 5px;
            resize: vertical;
        }
        .button-container {
            margin-top: 10px;
            display: flex;
            justify-content: flex-end;
        }
        #clearButton {
            padding: 8px 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        #clearButton:hover {
            background-color: #45a049;
        }
        #status {
            margin-top: 10px;
            color: #666;
        }
        .settings {
            margin-top: 20px;
            width: 100%;
            max-width: 800px;
        }
        .slider-container {
            margin-bottom: 15px;
        }
        .slider-container label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        .instructions {
            background-color: #e9f7ef;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
            width: 100%;
            max-width: 800px;
        }
        .instructions h3 {
            margin-top: 0;
            color: #2ecc71;
        }
        .instructions ol {
            margin: 0;
            padding-left: 20px;
        }
        .instructions li {
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Virtual Air Keyboard</h1>
        <p class="description">
            Type by making a pinch gesture (thumb and index finger) in front of your camera.
            The position of your pinch will determine which key is pressed.
        </p>
        
        <div class="video-text-container">
            <img id="videoElement" src="/video_feed">
            
            <div class="text-container">
                <textarea id="textOutput" readonly></textarea>
                <div class="button-container">
                    <button id="clearButton">Clear Text</button>
                </div>
                <div id="status">Ready</div>
            </div>
            
            <div class="settings">
                <h3>Settings</h3>
                <div class="slider-container">
                    <label for="cooldownSlider">Key Press Cooldown (seconds): <span id="cooldownValue">0.8</span></label>
                    <input type="range" id="cooldownSlider" min="0.2" max="2.0" step="0.1" value="0.8">
                </div>
            </div>
            
            <div class="instructions">
                <h3>How to Use</h3>
                <ol>
                    <li>Allow camera access when prompted</li>
                    <li>Position your hand in front of the camera</li>
                    <li>Make a pinch gesture (touch thumb and index finger) to 'press' keys</li>
                    <li>Hold the pinch briefly to register a key press</li>
                    <li>Move your hand to different positions to type different characters</li>
                </ol>
                <p><em>Note: This application works best with good lighting and a clear background.</em></p>
            </div>
        </div>
    </div>

    <script>
        // Poll for detected keys
        function pollDetectedKeys() {
            fetch('/get_detected_keys')
                .then(response => response.json())
                .then(data => {
                    if (data.keys && data.keys.length > 0) {
                        const textOutput = document.getElementById('textOutput');
                        let currentText = textOutput.value;
                        
                        // Process each detected key
                        data.keys.forEach(key => {
                            if (key === 'BACKSPACE' && currentText.length > 0) {
                                currentText = currentText.slice(0, -1);
                            } else if (key === 'SPACE') {
                                currentText += ' ';
                            } else if (key.length === 1) {
                                currentText += key;
                            }
                            
                            // Update status
                            document.getElementById('status').innerText = `Key pressed: ${key}`;
                        });
                        
                        textOutput.value = currentText;
                        
                        // Send the updated text back to the server
                        fetch('/update_text', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ text: currentText }),
                        });
                    }
                })
                .catch(error => {
                    console.error('Error polling for keys:', error);
                });
        }

        // Poll every 100ms
        setInterval(pollDetectedKeys, 100);

        // Clear button functionality
        document.getElementById('clearButton').addEventListener('click', function() {
            document.getElementById('textOutput').value = '';
            fetch('/update_text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: '' }),
            });
        });

        // Cooldown slider functionality
        document.getElementById('cooldownSlider').addEventListener('input', function(e) {
            const value = e.target.value;
            document.getElementById('cooldownValue').innerText = value;
            fetch('/set_cooldown', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ cooldown: parseFloat(value) }),
            });
        });
    </script>
</body>
</html>
    ''')

# Write the JavaScript for the app
with open('static/app.js', 'w') as f:
    f.write('''
// This file is intentionally empty as all JavaScript is included in the HTML file
    ''')

def draw_keyboard(frame):
    h, w, _ = frame.shape
    
    # Define keyboard dimensions
    keyboard_top = int(h * 0.6)
    keyboard_bottom = int(h * 0.95)
    keyboard_height = keyboard_bottom - keyboard_top
    
    # Calculate key dimensions based on layout
    max_keys_in_row = max(len(row) for row in keyboard_layout[:-1])  # Exclude the bottom row
    key_width = w // max_keys_in_row
    
    # Standard key height (for rows 0-3)
    std_key_height = keyboard_height // 5
    
    # Draw keys for standard rows (0-3)
    for row_idx, row in enumerate(keyboard_layout[:-1]):  # Process first 4 rows
        y_start = keyboard_top + row_idx * std_key_height
        y_end = y_start + std_key_height
        
        # Center the row if it has fewer keys than the max
        x_offset = int((w - len(row) * key_width) / 2)
        
        for key_idx, key in enumerate(row):
            x_start = x_offset + key_idx * key_width
            x_end = x_start + key_width
            
            # Draw key rectangle
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (200, 200, 200), 1)
            
            # Draw key text
            text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = x_start + (key_width - text_size[0]) // 2
            text_y = y_start + (std_key_height + text_size[1]) // 2
            cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw special keys (bottom row)
    y_start = keyboard_top + 4 * std_key_height
    y_end = keyboard_bottom
    
    # SPACE key (wider)
    space_width = w // 2
    cv2.rectangle(frame, (0, y_start), (space_width, y_end), (200, 200, 200), 1)
    text_size = cv2.getTextSize("SPACE", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = (space_width - text_size[0]) // 2
    text_y = y_start + (std_key_height + text_size[1]) // 2
    cv2.putText(frame, "SPACE", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # BACKSPACE key
    cv2.rectangle(frame, (space_width, y_start), (w, y_end), (200, 200, 200), 1)
    text_size = cv2.getTextSize("BACKSPACE", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = space_width + ((w - space_width) - text_size[0]) // 2
    text_y = y_start + (std_key_height + text_size[1]) // 2
    cv2.putText(frame, "BACKSPACE", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame, (keyboard_top, keyboard_bottom, key_width, std_key_height, space_width)

def get_key_at_position(x, y, keyboard_params):
    keyboard_top, keyboard_bottom, key_width, key_height, space_width = keyboard_params
    
    # Check if position is within keyboard area
    if y < keyboard_top or y > keyboard_bottom:
        return None
    
    # Calculate row index
    row_idx = min(int((y - keyboard_top) / key_height), 4)
    
    # Handle bottom row specially (SPACE and BACKSPACE)
    if row_idx == 4:
        if x < space_width:
            return "SPACE"
        else:
            return "BACKSPACE"
    
    # For standard rows, calculate column index
    # Center the keyboard row horizontally
    max_keys_in_row = len(keyboard_layout[row_idx])
    total_row_width = max_keys_in_row * key_width
    x_offset = int((1.0 - (total_row_width / 640)) * (640 / 2))
    
    # Adjust x to account for row centering
    adj_x = max(0, x - x_offset)
    col_idx = int(adj_x / key_width)
    
    # Check if the column index is valid for this row
    if col_idx >= 0 and col_idx < len(keyboard_layout[row_idx]):
        return keyboard_layout[row_idx][col_idx]
    
    return None

def check_pinch(hand_landmarks):
    # Get the positions of thumb tip and index finger tip
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Calculate the distance between thumb tip and index finger tip
    distance = np.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 + 
        (thumb_tip.y - index_tip.y) ** 2 + 
        (thumb_tip.z - index_tip.z) ** 2
    )
    
    # If the distance is small enough, consider it a pinch
    # The threshold may need to be adjusted
    pinch_threshold = 0.05
    is_pinching = distance < pinch_threshold
    
    # Calculate the midpoint between thumb and index finger (pinch position)
    pinch_x = (thumb_tip.x + index_tip.x) / 2
    pinch_y = (thumb_tip.y + index_tip.y) / 2
    
    return is_pinching, (pinch_x, pinch_y)

def process_frame():
    global last_key_press_time, key_cooldown, detected_keys
    
    # Initialize MediaPipe hands
    hands = mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1
    )
    
    while True:
        if cap is None or not cap.isOpened():
            # If camera is not available, provide a blank frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue
            
        success, frame = cap.read()
        if not success:
            # If frame capture failed, provide a blank frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue
        
        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Draw the keyboard overlay
        frame, keyboard_params = draw_keyboard(frame)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image with MediaPipe Hands
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks and check for pinch
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Check for pinch gesture
                is_pinching, (pinch_x, pinch_y) = check_pinch(hand_landmarks)
                
                # Convert normalized coordinates to pixel coordinates
                h, w, _ = frame.shape
                pinch_pixel_x = int(pinch_x * w)
                pinch_pixel_y = int(pinch_y * h)
                
                # Draw pinch point
                if is_pinching:
                    cv2.circle(frame, (pinch_pixel_x, pinch_pixel_y), 10, (0, 255, 0), -1)
                    
                    # Check time since last key press (to prevent multiple presses)
                    current_time = time.time()
                    if current_time - last_key_press_time > key_cooldown:
                        # Get the key at pinch position
                        key = get_key_at_position(pinch_pixel_x, pinch_pixel_y, keyboard_params)
                        
                        if key:
                            # Add key to detected keys list
                            with lock:
                                detected_keys.append(key)
                            
                            # Show visual feedback on the video
                            cv2.putText(frame, f"Key: {key}", (50, 50), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # Update the last key press time
                            last_key_press_time = current_time
                else:
                    # Draw a different color circle when not pinching
                    cv2.circle(frame, (pinch_pixel_x, pinch_pixel_y), 5, (255, 0, 0), -1)
        
        # Convert the frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        
        # Yield the frame in the multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(process_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detected_keys')
def get_detected_keys():
    """API endpoint to get detected keys."""
    global detected_keys
    with lock:
        keys = detected_keys.copy()
        detected_keys = []  # Clear the queue after it's been read
    return jsonify({'keys': keys})

@app.route('/update_text', methods=['POST'])
def update_text():
    """API endpoint to update text."""
    global current_text
    data = json.loads(request.data)
    current_text = data['text']
    return jsonify({'status': 'success'})

@app.route('/set_cooldown', methods=['POST'])
def set_cooldown():
    """API endpoint to set cooldown time."""
    global key_cooldown
    data = json.loads(request.data)
    key_cooldown = float(data['cooldown'])
    return jsonify({'status': 'success'})

def initialize_camera():
    """Initialize the camera."""
    global cap
    cap = cv2.VideoCapture(0)  # Use the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return False
    return True

if __name__ == '__main__':
    # Make sure we have the Flask dependencies
    try:
        from flask import request
    except ImportError:
        print("Flask dependencies missing. Install with: pip install flask")
        exit(1)
        
    # Initialize camera in a separate thread
    threading.Thread(target=initialize_camera).start()
    
    # Add a small delay to allow camera to initialize
    time.sleep(1)
    
    print("Starting Flask server...")
    print("Access the Air Keyboard at http://127.0.0.1:5000")
    app.run(debug=False, threaded=True)