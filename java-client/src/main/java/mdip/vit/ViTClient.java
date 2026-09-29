package mdip.vit;

import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

// used to POST to /vit/embed on the Python service, now runs vit.onnx
// locally through DJL and ONNX Runtime, same embedImage(File) signature
// so ViTIndexer and ViTSearcher did not have to change at all
public class ViTClient {

    private final Predictor<Image, float[]> predictor;

    public ViTClient(String modelsDir) throws Exception {
        Path modelPath = Paths.get(modelsDir);
        Criteria<Image, float[]> criteria = Criteria.builder()
                .setTypes(Image.class, float[].class)
                .optModelPath(modelPath)
                .optModelName("vit") // looks for modelsDir/vit.onnx
                .optEngine("OnnxRuntime")
                .optTranslator(new ViTTranslator())
                .build();
        ZooModel<Image, float[]> model = criteria.loadModel();
        predictor = model.newPredictor();
    }

    public double[] embedImage(File image) throws Exception {
        Image img = ImageFactory.getInstance().fromFile(image.toPath());
        float[] embedding = predictor.predict(img);
        double[] result = new double[embedding.length];
        for (int i = 0; i < embedding.length; i++) {
            result[i] = embedding[i];
        }
        return result;
    }

    // resize, tensor, normalize, same preprocessing ViTImageProcessor did
    // in the old Python service. mean/std/size come from export_models.py,
    // double check them against what that script prints
    private static class ViTTranslator implements Translator<Image, float[]> {
        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            NDArray array = input.toNDArray(ctx.getNDManager());
            array = new Resize(224, 224).transform(array);
            array = new ToTensor().transform(array);
            array = new Normalize(new float[]{0.5f, 0.5f, 0.5f}, new float[]{0.5f, 0.5f, 0.5f}).transform(array);
            // no manual expandDims here — Predictor.predict() already adds
            // the batch dimension for a single Image; doing it here too
            // produced a 5D tensor ([1,1,3,224,224]) instead of 4D
            return new NDList(array);
        }

        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            return list.singletonOrThrow().toFloatArray();
        }
    }
}