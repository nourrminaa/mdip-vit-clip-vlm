package mdip;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import mdip.clip.ClipClient;
import mdip.clip.ClipIndexer;
import mdip.clip.ClipSearcher;
import mdip.vit.ViTClient;
import mdip.vit.ViTIndexer;
import mdip.vit.ViTSearcher;
import mdip.vlm.VlmClient;
import mdip.vlm.VlmJsonStore;
import mdip.vlm.VlmResult;

public class Main {

    private static final String MODELS_DIR = "models";
    private static final File IMAGES_FOLDER = new File(ImageCatalog.IMAGES_DIR);
    private static final File VIT_INDEX_FILE = new File("vit-feature-vector.tsv");
    private static final File CLIP_INDEX_FILE = new File("clip-index.tsv");
    private static final File VLM_JSON_FILE = new File("vlm-results.json");

    private static final String GEMINI_MODEL = "gemini-3.5-flash-lite";
    // if a model is overloaded (503), the next attempt uses the next one
    private static final String[] GEMINI_FALLBACK_MODELS = {GEMINI_MODEL, "gemini-3.1-flash-lite", "gemini-3.6-flash"};

    // CLIP's learned temperature (logit_scale), cosine * 100 then softmax
    private static final double CLIP_LOGIT_SCALE = 100.0;

    // text to image queries
    private static final String[] TEXT_QUERIES = {
            "a dog",
            "a cat",
            "two people shaking hands",
            "people working on a laptop",
            "a whale jumping out of the water",
            "a car on a road",
            "a group of people standing together",
            "an animal on green grass",
    };

    // image to text candidates, CLIP picks the closest one
    private static final String[] CANDIDATE_TEXTS = {
            "a photo of a dog",
            "a photo of a cat",
            "a photo of two people shaking hands",
            "a photo of people working at a desk",
            "a photo of a whale jumping out of the water",
            "a photo of a car",
            "a photo of a group of people",
            "a photo of green grass",
    };

    // text to text candidates
    private static final String[] TEXT_TO_TEXT_CANDIDATES = {
            "two people greeting each other",
            "people shaking hands",
            "a puppy playing outside",
            "a kitten sleeping",
            "a whale jumping out of the water",
            "a car driving on a road",
            "a person sitting at a desk",
            "a business meeting",
    };

    // objects that are in none of the 8 images
    private static final String[] DISTRACTOR_OBJECTS = {"airplane", "pizza", "guitar", "elephant", "umbrella"};

    private static final Scanner scanner = new Scanner(System.in);

    // models only load the first time a menu needs them
    private static ViTClient vitClient;
    private static ClipClient clipClient;
    private static VlmClient vlmClient;

    public static void main(String[] args) throws Exception {
        while (true) {
            System.out.println();
            System.out.println("===== MDIP lab01 =====");
            System.out.println("1. ViT");
            System.out.println("2. CLIP");
            System.out.println("3. VLM (Gemini)");
            System.out.println("0. exit");
            int choice = readInt("choose: ");
            if (choice == 1) {
                vitMenu();
            } else if (choice == 2) {
                clipMenu();
            } else if (choice == 3) {
                vlmMenu();
            } else if (choice == 0) {
                return;
            } else {
                System.out.println("invalid choice!");
            }
        }
    }

    // ===================== ViT =====================

    private static void vitMenu() {
        while (true) {
            System.out.println();
            System.out.println("----- ViT -----");
            System.out.println("1. build ViT db (all images -> feature vectors -> " + VIT_INDEX_FILE.getName() + ")");
            System.out.println("2. show the feature vector of an image");
            System.out.println("3. cosine between 2 images");
            System.out.println("4. search the db with an image (top 5)");
            System.out.println("0. back");
            int choice = readInt("choose: ");
            if (choice == 0) {
                return;
            }
            try {
                if (choice == 1) {
                    buildVitDb();
                } else if (choice == 2) {
                    vitShowVector();
                } else if (choice == 3) {
                    vitCosine();
                } else if (choice == 4) {
                    vitSearch();
                } else {
                    System.out.println("invalid choice");
                }
            } catch (Exception e) {
                System.out.println("error: " + e.getMessage());
            }
        }
    }

    // the "db": one line per image, path tab feature vector (768 values)
    private static void buildVitDb() throws Exception {
        new ViTIndexer(getVit(), VIT_INDEX_FILE).createIndex(IMAGES_FOLDER);
        System.out.println("ViT db saved to " + VIT_INDEX_FILE.getName());
    }

    private static void vitShowVector() throws Exception {
        File image = pickImage("pick an image:");
        if (image == null) {
            return;
        }
        double[] vector = getVit().embedImage(image);
        System.out.println(ImageCatalog.labelFor(image.getPath()) + " -> " + VectorUtil.preview(vector, 10));
    }

    private static void vitCosine() throws Exception {
        File first = pickImage("pick the first image:");
        if (first == null) {
            return;
        }
        File second = pickImage("pick the second image:");
        if (second == null) {
            return;
        }
        double[] firstVector = getVit().embedImage(first);
        double[] secondVector = getVit().embedImage(second);
        System.out.println(ImageCatalog.labelFor(first.getPath()) + " -> " + VectorUtil.preview(firstVector, 10));
        System.out.println(ImageCatalog.labelFor(second.getPath()) + " -> " + VectorUtil.preview(secondVector, 10));
        double score = VectorUtil.cosineSimilarity(firstVector, secondVector);
        System.out.println("ViT cosine similarity: " + format(score));
    }

    private static void vitSearch() throws Exception {
        if (!VIT_INDEX_FILE.exists()) {
            buildVitDb();
        }
        File query = pickImage("pick the query image:");
        if (query == null) {
            return;
        }
        double[] queryVector = getVit().embedImage(query);
        System.out.println("ViT similar images:");
        for (ViTSearcher.Result result : new ViTSearcher(VIT_INDEX_FILE).search(queryVector, 5)) {
            System.out.println("  " + ImageCatalog.labelFor(result.imagePath) + "  " + format(result.score) + sameImageTag(query, result.imagePath));
        }
    }

    // ===================== CLIP =====================

    private static void clipMenu() {
        while (true) {
            System.out.println();
            System.out.println("----- CLIP -----");
            System.out.println("1. build CLIP index (all images -> feature vectors -> " + CLIP_INDEX_FILE.getName() + ")");
            System.out.println("2. text to image");
            System.out.println("3. image to text");
            System.out.println("4. text to text");
            System.out.println("5. image to image (search the index)");
            System.out.println("6. similarity between 2 images (grass test: dog vs cat)");
            System.out.println("0. back");
            int choice = readInt("choose: ");
            if (choice == 0) {
                return;
            }
            try {
                if (choice == 1) {
                    buildClipIndex();
                } else if (choice == 2) {
                    clipTextToImage();
                } else if (choice == 3) {
                    clipImageToText();
                } else if (choice == 4) {
                    clipTextToText();
                } else if (choice == 5) {
                    clipImageToImage();
                } else if (choice == 6) {
                    clipTwoImages();
                } else {
                    System.out.println("invalid choice");
                }
            } catch (Exception e) {
                System.out.println("error: " + e.getMessage());
            }
        }
    }

    // step 1 of every CLIP search: all images -> feature vectors, done once
    private static void buildClipIndex() throws Exception {
        new ClipIndexer(getClip(), CLIP_INDEX_FILE).createIndex(IMAGES_FOLDER);
        System.out.println("CLIP index saved to " + CLIP_INDEX_FILE.getName());
    }

    // CLIP text to image: text -> tokenizer -> text vector, cosine against
    // every image vector in the index, softmax turns the scores into probabilities
    private static void clipTextToImage() throws Exception {
        if (!CLIP_INDEX_FILE.exists()) {
            buildClipIndex();
        }
        String text = pickText("text to image, pick a text:", TEXT_QUERIES);
        if (text == null) {
            return;
        }
        printTokens(text);
        double[] textVector = getClip().embedText(text);
        List<ClipSearcher.Result> results = new ClipSearcher(CLIP_INDEX_FILE).searchByVector(textVector, Integer.MAX_VALUE);

        String[] names = new String[results.size()];
        double[] scores = new double[results.size()];
        for (int i = 0; i < results.size(); i++) {
            names[i] = ImageCatalog.labelFor(results.get(i).imagePath);
            scores[i] = results.get(i).score;
        }
        System.out.println("CLIP text to image matches (cosine | softmax probability):");
        printRanked(names, scores, VectorUtil.softmax(scores, CLIP_LOGIT_SCALE));
    }

    // CLIP image to text: CLIP does not generate text, so the image is compared against candidate texts
    private static void clipImageToText() throws Exception {
        File image = pickImage("image to text, pick an image:");
        if (image == null) {
            return;
        }
        List<String> texts = new ArrayList<>(Arrays.asList(CANDIDATE_TEXTS));
        System.out.print("add your own candidate text too? (enter to skip): ");
        String extra = scanner.nextLine().trim();
        if (!extra.isEmpty()) {
            texts.add(extra);
        }

        double[] imageVector = getClip().embedImage(image);
        String[] names = texts.toArray(new String[0]);
        double[] scores = new double[names.length];
        for (int i = 0; i < names.length; i++) {
            scores[i] = VectorUtil.cosineSimilarity(imageVector, getClip().embedText(names[i]));
        }
        System.out.println("CLIP image to text matches for " + ImageCatalog.labelFor(image.getPath()) + " (cosine | softmax probability):");
        printRanked(names, scores, VectorUtil.softmax(scores, CLIP_LOGIT_SCALE));
    }

    // CLIP text to text: both go through the text tower. no softmax here,
    // CLIP was trained to match text with images, not text with text, so
    // the raw cosine is the honest number to show
    private static void clipTextToText() throws Exception {
        String query = pickText("text to text, pick the query text:", TEXT_QUERIES);
        if (query == null) {
            return;
        }
        printTokens(query);
        double[] queryVector = getClip().embedText(query);
        double[] scores = new double[TEXT_TO_TEXT_CANDIDATES.length];
        for (int i = 0; i < TEXT_TO_TEXT_CANDIDATES.length; i++) {
            scores[i] = VectorUtil.cosineSimilarity(queryVector, getClip().embedText(TEXT_TO_TEXT_CANDIDATES[i]));
        }
        System.out.println("CLIP text to text matches (cosine):");
        printRanked(TEXT_TO_TEXT_CANDIDATES, scores, null);
    }

    // CLIP image to image: query image vector against the whole index
    private static void clipImageToImage() throws Exception {
        if (!CLIP_INDEX_FILE.exists()) {
            buildClipIndex();
        }
        File query = pickImage("image to image, pick the query image:");
        if (query == null) {
            return;
        }
        double[] queryVector = getClip().embedImage(query);
        System.out.println("CLIP image to image matches:");
        for (ClipSearcher.Result result : new ClipSearcher(CLIP_INDEX_FILE).searchByVector(queryVector, 5)) {
            System.out.println("  " + ImageCatalog.labelFor(result.imagePath) + "  " + format(result.score) + sameImageTag(query, result.imagePath));
        }
    }

    // similarity between 2 images, then check what they share (e.g. grass):
    // if both images score high against "grass", that's part of why they're similar
    private static void clipTwoImages() throws Exception {
        File first = pickImage("pick the first image:");
        if (first == null) {
            return;
        }
        File second = pickImage("pick the second image:");
        if (second == null) {
            return;
        }
        double[] firstVector = getClip().embedImage(first);
        double[] secondVector = getClip().embedImage(second);
        System.out.println("CLIP cosine between " + ImageCatalog.labelFor(first.getPath()) + " and "
                + ImageCatalog.labelFor(second.getPath()) + ": " + format(VectorUtil.cosineSimilarity(firstVector, secondVector)));

        System.out.print("what do they share? type a word like grass (enter to skip): ");
        String shared = scanner.nextLine().trim();
        if (shared.isEmpty()) {
            return;
        }
        double[] sharedVector = getClip().embedText("a photo of " + shared);
        System.out.println("  " + ImageCatalog.labelFor(first.getPath()) + " vs \"" + shared + "\": " + format(VectorUtil.cosineSimilarity(firstVector, sharedVector)));
        System.out.println("  " + ImageCatalog.labelFor(second.getPath()) + " vs \"" + shared + "\": " + format(VectorUtil.cosineSimilarity(secondVector, sharedVector)));
    }

    // ===================== VLM =====================

    private static void vlmMenu() {
        while (true) {
            System.out.println();
            System.out.println("----- VLM (Gemini " + GEMINI_MODEL + ") -----");
            System.out.println("1. describe an image (objects, actions, relationships) -> " + VLM_JSON_FILE.getName());
            System.out.println("2. describe all 8 images -> " + VLM_JSON_FILE.getName());
            System.out.println("3. show " + VLM_JSON_FILE.getName());
            System.out.println("0. back");
            int choice = readInt("choose: ");
            if (choice == 0) {
                return;
            }
            try {
                if (choice == 1) {
                    vlmDescribeOne();
                } else if (choice == 2) {
                    vlmDescribeAll();
                } else if (choice == 3) {
                    vlmShowJson();
                } else {
                    System.out.println("invalid choice");
                }
            } catch (Exception e) {
                System.out.println("error: " + e.getMessage());
            }
        }
    }

    private static void vlmDescribeOne() throws Exception {
        File image = pickImage("pick an image:");
        if (image == null) {
            return;
        }
        VlmResult result = describeWithRetry(image);
        if (result == null) {
            return;
        }
        new VlmJsonStore(VLM_JSON_FILE).save(result);
        printVlmResult(result);
        System.out.println("saved to " + VLM_JSON_FILE.getName());
    }

    private static void vlmDescribeAll() throws Exception {
        VlmJsonStore store = new VlmJsonStore(VLM_JSON_FILE);
        System.out.print("redo images already in " + VLM_JSON_FILE.getName() + "? (y/n): ");
        boolean redo = scanner.nextLine().trim().equalsIgnoreCase("y");
        for (int i = 0; i < ImageCatalog.count(); i++) {
            File image = ImageCatalog.getFile(i);
            if (!image.exists()) {
                System.out.println("skipping " + ImageCatalog.LABELS[i] + ", missing " + image.getPath());
                continue;
            }
            if (!redo && store.get(image.getName()) != null) {
                System.out.println("already done: " + ImageCatalog.LABELS[i]);
                continue;
            }
            System.out.println("describing " + ImageCatalog.LABELS[i] + "...");
            VlmResult result = describeWithRetry(image);
            if (result == null) {
                continue;
            }
            store.save(result);
            printVlmResult(result);
            // free tier has a requests-per-minute limit, small pause between calls
            Thread.sleep(4000);
        }
        System.out.println("all saved to " + VLM_JSON_FILE.getName());
    }

    // one failed call (timeout, 429, 503) shouldn't kill the whole loop,
    // every attempt switches to the next model in case one is overloaded
    private static VlmResult describeWithRetry(File image) throws Exception {
        for (int attempt = 0; attempt < GEMINI_FALLBACK_MODELS.length; attempt++) {
            String model = GEMINI_FALLBACK_MODELS[attempt];
            try {
                VlmResult result = getVlm().describeImage(image, model);
                System.out.println("  (answered by " + model + ")");
                return result;
            } catch (Exception e) {
                System.out.println("  " + model + " failed: " + shortError(e));
                Thread.sleep(5000);
            }
        }
        System.out.println("  giving up on " + image.getName() + ", run option 2 again later");
        return null;
    }

    // the full Gemini error JSON is noise, the first line is enough
    private static String shortError(Exception e) {
        String message = String.valueOf(e.getMessage());
        return message.length() > 90 ? message.substring(0, 90) + "..." : message;
    }

    private static void vlmShowJson() throws Exception {
        if (!VLM_JSON_FILE.exists()) {
            System.out.println(VLM_JSON_FILE.getName() + " doesn't exist yet, describe an image first");
            return;
        }
        System.out.println(Files.readString(VLM_JSON_FILE.toPath()));
    }

    private static void printVlmResult(VlmResult result) {
        System.out.println("Image: " + result.image);
        System.out.println("Objects: " + (result.objects == null ? "" : String.join(", ", result.objects)));
        System.out.println("Actions:");
        if (result.actions != null) {
            for (VlmResult.Action action : result.actions) {
                System.out.println("  " + action.object + " -> " + action.action);
            }
        }
        System.out.println("Relationships:");
        if (result.relationships != null) {
            for (VlmResult.Relationship relationship : result.relationships) {
                System.out.println("  " + relationship.subject + " | " + relationship.relation + " | " + relationship.object);
            }
        }
        System.out.println("Description: " + result.description);
    }

    // ===================== helpers =====================

    private static ViTClient getVit() throws Exception {
        if (vitClient == null) {
            System.out.println("loading ViT...");
            vitClient = new ViTClient(MODELS_DIR);
        }
        return vitClient;
    }

    private static ClipClient getClip() throws Exception {
        if (clipClient == null) {
            System.out.println("loading CLIP...");
            clipClient = new ClipClient(MODELS_DIR);
        }
        return clipClient;
    }

    private static VlmClient getVlm() {
        if (vlmClient == null) {
            String apiKey = System.getenv("GEMINI_API_KEY");
            if (apiKey == null || apiKey.isBlank()) {
                throw new RuntimeException("GEMINI_API_KEY is not set;");
            }
            vlmClient = new VlmClient(apiKey, GEMINI_MODEL);
        }
        return vlmClient;
    }

    // numbered list of the 8 images, returns the chosen file or null
    private static File pickImage(String title) {
        System.out.println(title);
        for (int i = 0; i < ImageCatalog.count(); i++) {
            String missing = ImageCatalog.getFile(i).exists() ? "" : "  (missing: images/" + ImageCatalog.FILES[i] + ")";
            System.out.println((i + 1) + ". " + ImageCatalog.LABELS[i] + missing);
        }
        int choice = readInt("choose: ");
        if (choice < 1 || choice > ImageCatalog.count()) {
            System.out.println("invalid choice");
            return null;
        }
        File image = ImageCatalog.getFile(choice - 1);
        if (!image.exists()) {
            System.out.println("put the image at " + image.getPath() + " first");
            return null;
        }
        return image;
    }

    // preset texts + "other" at the end to type your own
    private static String pickText(String title, String[] options) {
        System.out.println(title);
        for (int i = 0; i < options.length; i++) {
            System.out.println((i + 1) + ". " + options[i]);
        }
        System.out.println((options.length + 1) + ". other (type your own)");
        int choice = readInt("choose: ");
        if (choice == options.length + 1) {
            System.out.print("type your text: ");
            String text = scanner.nextLine().trim();
            return text.isEmpty() ? null : text;
        }
        if (choice < 1 || choice > options.length) {
            System.out.println("invalid choice");
            return null;
        }
        return options[choice - 1];
    }

    private static void printTokens(String text) throws Exception {
        System.out.println("tokens CLIP sees for \"" + text + "\": " + String.join(" ", getClip().tokenize(text)));
    }

    // prints names sorted by score, highest first, probabilities optional
    private static void printRanked(String[] names, double[] scores, double[] probabilities) {
        Integer[] order = new Integer[names.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(scores[b], scores[a]));
        for (int i : order) {
            String line = "  " + names[i] + "  " + format(scores[i]);
            if (probabilities != null) {
                line += " | " + String.format("%.2f%%", probabilities[i] * 100);
            }
            System.out.println(line);
        }
    }

    private static String sameImageTag(File query, String resultPath) {
        return query.getName().equals(new File(resultPath).getName()) ? "  (itself)" : "";
    }

    private static String format(double value) {
        return String.format("%.4f", value);
    }

    private static int readInt(String prompt) {
        System.out.print(prompt);
        String line = scanner.nextLine().trim();
        try {
            return Integer.parseInt(line);
        } catch (NumberFormatException e) {
            return -1;
        }
    }
}
