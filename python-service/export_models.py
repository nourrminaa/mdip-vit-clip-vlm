# One time export script. Converts the ViT and CLIP PyTorch models into ONNX
# files that the Java client loads directly through DJL, no server, no ngrok
# needed for these two anymore. Run this once wherever you have internet
# access to pull the pretrained weights (Colab is fine, same as app.py),
# then copy everything from exported-models/ into java-client/models/.
#
#   pip install torch transformers
#   python export_models.py

import os
import torch
from transformers import ViTImageProcessor, ViTModel
from transformers import CLIPProcessor, CLIPModel

OUT_DIR = "exported-models"
os.makedirs(OUT_DIR, exist_ok=True)


# wraps ViTModel so the onnx graph outputs the CLS embedding directly,
# same 768-dim vector the old /vit/embed endpoint returned
class VitWrapper(torch.nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit_model = vit_model

    def forward(self, pixel_values):
        outputs = self.vit_model(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0, :]


# wraps CLIP's image tower so the onnx graph is just image in, vector out
class ClipVisionWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, pixel_values):
        return self.clip_model.get_image_features(pixel_values=pixel_values)


# wraps CLIP's text tower so the onnx graph is tokens in, vector out
class ClipTextWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, input_ids, attention_mask):
        return self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)


def export_vit():
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").eval()
    wrapper = VitWrapper(model)

    size = processor.size["height"]
    dummy = torch.zeros(1, 3, size, size)
    torch.onnx.export(
        wrapper, dummy, f"{OUT_DIR}/vit.onnx",
        input_names=["pixel_values"], output_names=["embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
    # ViTTranslator.java hardcodes mean/std/size, double check they match these
    print("ViT  mean:", processor.image_mean, "std:", processor.image_std, "size:", size)


def export_clip():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()

    vision = ClipVisionWrapper(model)
    image_size = model.config.vision_config.image_size
    dummy_image = torch.zeros(1, 3, image_size, image_size)
    torch.onnx.export(
        vision, dummy_image, f"{OUT_DIR}/clip_vision.onnx",
        input_names=["pixel_values"], output_names=["embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )

    text = ClipTextWrapper(model)
    dummy_ids = torch.zeros(1, 16, dtype=torch.long)
    dummy_mask = torch.ones(1, 16, dtype=torch.long)
    torch.onnx.export(
        text, (dummy_ids, dummy_mask), f"{OUT_DIR}/clip_text.onnx",
        input_names=["input_ids", "attention_mask"], output_names=["embedding"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "embedding": {0: "batch"},
        },
        opset_version=17,
    )

    # saves tokenizer.json, the file DJL's HuggingFace tokenizer library reads
    processor.tokenizer.save_pretrained(OUT_DIR)
    # ClipClient.java hardcodes mean/std/size, double check they match these
    print("CLIP image_mean:", processor.image_processor.image_mean,
          "image_std:", processor.image_processor.image_std,
          "size:", image_size)


if __name__ == "__main__":
    export_vit()
    export_clip()
    print("done, copy everything in", OUT_DIR, "into java-client/models/")
