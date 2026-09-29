# FastAPI Server that exposes the VLM (BLIP-2) as an HTTP endpoint
# reachable to my local machine (client) through ngrok public tunneling: tunnels the localhost:8000 to "https://baked-variable-disallow.ngrok-free.dev"
# ViT and CLIP used to live here too, they now run locally in the Java client
# through DJL and ONNX Runtime, see java-client/models and export_models.py

import io
import threading
import torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import PlainTextResponse
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import nest_asyncio
import uvicorn
from pyngrok import ngrok

app = FastAPI()
device = "cpu"

# models loading
blip_dtype = torch.float16
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=blip_dtype).to(device).eval()


# bytes to RGB
def load_image(file_bytes):
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

# runs BLIP-2 with a given prompt & returns c
def ask_blip(image, prompt):
    inputs = blip_processor(images=image, text=prompt, return_tensors="pt").to(device, blip_dtype)
    with torch.no_grad():
        generated = blip_model.generate(**inputs, max_new_tokens=100)
    return blip_processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

# VLM: image to plain text block
# 3 questions are being asked to describe
@app.post("/vlm/describe", response_class=PlainTextResponse)
async def vlm_describe(file: UploadFile = File(...)):
    image = load_image(await file.read())
    objects = ask_blip(image, "Question: List every object visible in the image, separated by commas. Answer:")
    relationships = ask_blip(image, "Question: Describe the spatial relationships between the objects in the image, such as in front of, behind, on top of, below, left of and right of. Answer:")
    description = ask_blip(image, "Question: Describe this image in detail. Answer:")
    return "OBJECTS:\n" + objects + "\nRELATIONSHIPS:\n" + relationships + "\nDESCRIPTION:\n" + description


nest_asyncio.apply() 
ngrok.set_auth_token("3JejUkeLLtOt7nk8NZrIy3Jt3SK_VqHHp123tvc2ZwtGrumV")
# opens a public tunnel to the local server
public_url = ngrok.connect(8000)
print(public_url)

config = uvicorn.Config(app=app, host="0.0.0.0", port=8000)
server = uvicorn.Server(config)
await server.serve()
