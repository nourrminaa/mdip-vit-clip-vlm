package mdip.clip;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

// used to POST to /clip/embed-image and /clip/embed-text on the Python
// service, now runs clip_vision.onnx and clip_text.onnx locally through
// DJL and ONNX Runtime, same embedImage(File)/embedText(String) signatures
// so ClipIndexer and ClipSearcher did not have to change at all
public class ClipClient {

    private final Predictor<Image, float[]> imagePredictor;
    private final Predictor<String, float[]> textPredictor;
    private final HuggingFaceTokenizer tokenizer;

    public ClipClient(String modelsDir) throws Exception {
        Path dir = Paths.get(modelsDir);

        Criteria<Image, float[]> imageCriteria = Criteria.builder()
                .setTypes(Image.class, float[].class)
                .optModelPath(dir)
                .optModelName("clip_vision") // looks for modelsDir/clip_vision.onnx
                .optEngine("OnnxRuntime")
                .optTranslator(new ClipImageTranslator())
                .build();
        ZooModel<Image, float[]> imageModel = imageCriteria.loadModel();
        imagePredictor = imageModel.newPredictor();

        // same tokenizer.json CLIPProcessor saved in export_models.py
        tokenizer = HuggingFaceTokenizer.newInstance(dir.resolve("tokenizer.json"));
        Criteria<String, float[]> textCriteria = Criteria.builder()
                .setTypes(String.class, float[].class)
                .optModelPath(dir)
                .optModelName("clip_text") // looks for modelsDir/clip_text.onnx
                .optEngine("OnnxRuntime")
                .optTranslator(new ClipTextTranslator(tokenizer))
                .build();
        ZooModel<String, float[]> textModel = textCriteria.loadModel();
        textPredictor = textModel.newPredictor();
    }

    public double[] embedImage(File image) throws Exception {
        Image img = ImageFactory.getInstance().fromFile(image.toPath());
        return toDoubleArray(imagePredictor.predict(img));
    }

    public double[] embedText(String text) throws Exception {
        return toDoubleArray(textPredictor.predict(text));
    }

    // shows what the text tower actually receives: the text split into
    // CLIP tokens (words/sub-words, "</w>" marks the end of a word) and
    // the id of each one, padding tokens are skipped with the attention mask
    public List<String> tokenize(String text) {
        Encoding encoding = tokenizer.encode(text);
        String[] tokens = encoding.getTokens();
        long[] ids = encoding.getIds();
        long[] mask = encoding.getAttentionMask();
        List<String> result = new ArrayList<>();
        for (int i = 0; i < tokens.length; i++) {
            if (mask[i] == 1) {
                result.add(tokens[i] + "(" + ids[i] + ")");
            }
        }
        return result;
    }

    private double[] toDoubleArray(float[] values) {
        double[] result = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = values[i];
        }
        return result;
    }

    // resize, tensor, normalize with CLIP's own mean and std, not ViT's.
    // values come from export_models.py, double check them against what
    // that script prints
    private static class ClipImageTranslator implements Translator<Image, float[]> {
        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            NDArray array = input.toNDArray(ctx.getNDManager());
            array = new Resize(224, 224).transform(array);
            array = new ToTensor().transform(array);
            array = new Normalize(
                    new float[]{0.48145466f, 0.4578275f, 0.40821073f},
                    new float[]{0.26862954f, 0.26130258f, 0.27577711f}
            ).transform(array);
            // no manual expandDims — Predictor.predict() already batches
            // a single Image; doing it here too produced rank 5 instead of 4
            return new NDList(array);
        }

        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            return list.singletonOrThrow().toFloatArray();
        }
    }

    // tokenizes with tokenizer.json, feeds input_ids and attention_mask,
    // same input names ClipTextWrapper exported the onnx graph with
    private static class ClipTextTranslator implements Translator<String, float[]> {
        private final HuggingFaceTokenizer tokenizer;

        ClipTextTranslator(HuggingFaceTokenizer tokenizer) {
            this.tokenizer = tokenizer;
        }

        @Override
        public NDList processInput(TranslatorContext ctx, String input) {
            Encoding encoding = tokenizer.encode(input);
            // no manual expandDims here either — same double-batching issue
            // as the image translator, just for rank 2 (input_ids/attention_mask)
            // instead of rank 4
            NDArray inputIds = ctx.getNDManager().create(encoding.getIds()).toType(DataType.INT64, false);
            NDArray attentionMask = ctx.getNDManager().create(encoding.getAttentionMask()).toType(DataType.INT64, false);
            return new NDList(inputIds, attentionMask);
        }

        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            return list.singletonOrThrow().toFloatArray();
        }
    }
}