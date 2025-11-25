"""Simple WebSocket client to test the crowd management API."""

import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

async def test_websocket():
    """Test WebSocket connection and display frames."""
    uri = "ws://localhost:8000/ws/crowd"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket server")
            print("Receiving frames... (Press Ctrl+C to stop)")
            
            frame_count = 0
            while True:
                # Receive message
                message = await websocket.recv()
                data = json.loads(message)
                
                # Decode image
                image_data = base64.b64decode(data["image"])
                nparr = np.frombuffer(image_data, dtype=np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Display stats on frame
                    stats = data.get("stats", {})
                    count = stats.get("count", 0)
                    density = stats.get("density_score", 0.0)
                    
                    cv2.putText(frame, f"Count: {count}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Density: {density:.2f}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow("Crowd Detection", frame)
                    frame_count += 1
                    
                    if frame_count % 30 == 0:
                        print(f"Frames received: {frame_count}, Stats: {stats}")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(test_websocket())

