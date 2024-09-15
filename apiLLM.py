import asyncio
import websockets
import json
import boto3
from datetime import datetime
from botocore.client import Config
from computerVision import apply_model_to_video

# CLOUDFLARE CONFIG
ACCESS_KEY_ID = 'b19d514f16f18783ff2a290c04f619be'
SECRET_ACCESS_KEY = 'bd6a46cbdf3fd63f718caba171cd8fd5ecd917db709246499dc60371b935d31d'
BUCKET_NAME = 'hackmty'
REGION_NAME = 'auto'  # Use 'auto' as the region name for Cloudflare
ENDPOINT_URL = 'https://6619572a94d5a1d039dc239d940199ed.r2.cloudflarestorage.com'

async def getHeatmaps(websocket, path):
    # <LOGICA DE OBTENCION DE HEATMAPS>
    
    # <CLOUDFLARE>
    session = boto3.session.Session()

    s3_client = session.client(
        's3',
        region_name=REGION_NAME,
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4')  # R2 uses S3v4 signature
    )

    # Function to upload the file and get its public URL
    def upload_file_to_r2(file_path, object_name=None):
        if object_name is None:
            object_name = file_path

        try:
            # Upload the file
            s3_client.upload_file(file_path, BUCKET_NAME, object_name)

            # Generate the public URL (assuming the file is publicly accessible)
            public_url = f"{ENDPOINT_URL}/{BUCKET_NAME}/{object_name}"

            return public_url

        except Exception as e:
            print(f"Error uploading file: {e}")
            return None

    async for message in websocket:
        url = message
        
        video_output_path = 'video_output.mp4'
        heatmap_path = 'heatmap.jpg'
        last_frame_path = 'last_frame.jpg'
        
        apply_model_to_video(url, video_output_path, heatmap_path, last_frame_path)
        
        video_url = upload_file_to_r2(video_output_path, "video" + datetime.now().strftime("%Y%m%d%H%M%S") + ".mp4")
        heatmap_url = upload_file_to_r2(heatmap_path, "heatmap" + datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg")
        last_frame_url = upload_file_to_r2(last_frame_path, "last_frame" + datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg")
        file_urls = [video_url, last_frame_url]

        await websocket.send(json.dumps(file_urls))        

async def echo(websocket, path):
    if path == '/heatmaps':
        await getHeatmaps(websocket, path)
    else:
        await websocket.send("Unknown path")

async def runApiLLM():
    async with websockets.serve(echo, "localhost", 8765):
        print("WebSocket server running on ws://localhost:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(runApiLLM())