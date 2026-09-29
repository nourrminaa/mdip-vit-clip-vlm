# MDIP: ViT, CLIP and VLM in Java

ViT and CLIP run directly inside the Java client through DJL and ONNX Runtime.
The VLM is Gemini, called straight from Java over its REST API (the first
version used BLIP-2 on Colab behind ngrok, replaced because it was too heavy
and unstable). Everything is driven from one terminal menu so each model can
be tested separately.

## Models

| Model | Hugging Face / API | Size | Output |
|---|---|---|---|
| ViT | `google/vit-base-patch16-224-in21k` | ~86M params, ~340 MB | 768-d image vector ([CLS] token) |
| CLIP | `openai/clip-vit-base-patch32` | ~151M params, ~600 MB | 512-d image and text vectors (same space) |
| VLM | Gemini API (`gemini-3.6-flash` + fallbacks) | runs in the cloud | JSON: objects, actions, spatial relations |

## One-time setup: exporting ViT and CLIP

The `.onnx` files are not in the repo (too big for GitHub), you have to export them once.

1. Wherever you have internet access (Colab is fine), run:
   ```
   cd python-service
   pip install torch transformers
   python export_models.py
   ```
   This writes `vit.onnx`, `clip_vision.onnx`, `clip_text.onnx` and
   `tokenizer.json` into `exported-models/`.
2. Copy those four files into `java-client/models/`.
3. Check the mean/std/size values the script printed against the ones in
   `ViTClient.java` and `ClipClient.java`. They already match for the default models.

## Images

The 9 test images go in `java-client/images/`. The menu shows them by label,
see `ImageCatalog.java` to rename or add images.

| # | label | file |
|---|-------|------|
| 1 | dog image (on grass) | dog.jpeg |
| 2 | cat image (on grass) | cat.jpeg |
| 3 | handshake image 1 (colour) | handshake.jpg |
| 4 | handshake image 2 (black & white) | handshake-2.jpg |
| 5 | handshake image 3 (colour) | handshake-3.jpg |
| 6 | teamwork image | teamwork.jpg |
| 7 | whale image | whale.jpg |
| 8 | car image | car.jpeg |
| 9 | group of people image | group.jpeg |

## VLM setup (Gemini, free tier)

1. Get a free key at https://aistudio.google.com/apikey
2. `export GEMINI_API_KEY=your_key` (or add it to the IntelliJ run
   configuration's environment variables).
3. Main model is `GEMINI_MODEL` in `Main.java` (`gemini-3.6-flash`). If it's
   overloaded (503) the next one in `GEMINI_FALLBACK_MODELS` is tried
   (`gemini-3.5-flash-lite`, then `gemini-3.1-flash-lite`).

## Running

Needs Java 17+ and Maven.

```
cd java-client
mvn compile exec:java -Dexec.mainClass=mdip.Main
```

First run downloads the dependencies (DJL, ONNX Runtime, PyTorch native, ~200 MB).

Menu:

- **ViT**: build the ViT db (`vit-feature-vector.tsv`), show an image's
  feature vector, cosine between 2 images, search the db with an image (top 5).
- **CLIP**: build the index (`clip-index.tsv`), text to image, image to text,
  text to text, image to image, similarity between 2 images + a "what do they
  share" check (e.g. dog vs cat with "grass"). Text inputs have preset options
  and an "other" option to type your own, and the CLIP tokens are printed for
  every text. Text to image and image to text also show softmax probabilities
  (cosine x 100, CLIP's logit scale, then softmax).
- **VLM**: describe one image or all of them (images already saved are skipped
  unless you choose to redo them), or show `vlm-results.json`, which maps every
  image to its objects, what they're doing (actions) and their spatial
  relationships.

## Code structure

- `mdip.Main`: the menu, loads each model only the first time it's needed.
- `mdip.ImageCatalog`: menu labels to image files.
- `mdip.VectorUtil`: cosine similarity and softmax.
- `mdip.HttpUtil`: HTTP requests (used for Gemini).
- `mdip.vit`: ViTClient, ViTIndexer, ViTSearcher.
- `mdip.clip`: ClipClient, ClipIndexer, ClipSearcher.
- `mdip.vlm`: VlmClient (Gemini call), VlmResult, VlmJsonStore (`vlm-results.json`).

Indexes are flat TSV files (`path` tab `vector`) searched by brute force
cosine similarity, same Indexer/Searcher structure as LAB1 but with a learned
embedding instead of a colour histogram. They're built automatically the first
time a search needs them.

## Swapping models

If you swap ViT or CLIP for a different checkpoint, re-run `export_models.py`
with the new model name and copy the new files into `java-client/models/`.
No Java changes needed as long as the embedding size stays the same.