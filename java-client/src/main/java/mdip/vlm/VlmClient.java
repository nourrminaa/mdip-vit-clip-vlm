package mdip.vlm;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.File;
import java.nio.file.Files;
import java.util.Base64;
import mdip.HttpUtil;

// used to POST the image to BLIP-2 on the Python/Colab service through ngrok,
// now calls the Gemini API straight from Java: image goes in as base64,
// JSON comes back (objects, actions, relationships, description).
// same describeImage(File) signature as before
public class VlmClient {

    private static final String GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/";

    // one prompt instead of BLIP-2's 3 separate questions, and it asks for
    // JSON directly so there's no OBJECTS:/RELATIONSHIPS: text to cut up anymore
    private static final String PROMPT =
            "Look at this image and answer ONLY with a JSON object in this exact shape:\n"
            + "{\"objects\": [\"...\"],"
            + " \"actions\": [{\"object\": \"...\", \"action\": \"...\"}],"
            + " \"relationships\": [{\"subject\": \"...\", \"relation\": \"...\", \"object\": \"...\"}],"
            + " \"description\": \"...\"}\n"
            + "Rules:\n"
            + "- objects: every visible object, lowercase singular nouns. If there are several of the same thing, number them (person 1, person 2).\n"
            + "- actions: what each person or animal is doing, e.g. {\"object\": \"person 1\", \"action\": \"shaking hands with person 2\"}. Skip objects that are not doing anything.\n"
            + "- relationships: each one must read as a TRUE sentence 'subject relation object' from left to right,"
            + " e.g. {\"subject\": \"person 2\", \"relation\": \"holding\", \"object\": \"pen\"}, never the reverse."
            + " relation must be one of: left of, right of, in front of, behind, on top of, below, next to, inside, holding, wearing.\n"
            + "- give every object at least one relationship.\n"
            + "- for body parts, name the owner (\"hand of person 1\"), not \"hand 1\".\n"
            + "- use exactly the same object names in actions and relationships as in objects.\n"
            + "- use exactly the same object names in actions and relationships as in objects.\n"
            + "- description: 1 to 2 sentences describing the whole image.";

    private final String apiKey;
    private final String model;
    private final Gson gson = new Gson();

    public VlmClient(String apiKey, String model) {
        this.apiKey = apiKey;
        this.model = model;
    }

    public VlmResult describeImage(File image) throws Exception {
        return describeImage(image, model);
    }

    // same call, but with a specific model, used for fallback when one is overloaded (503)
    public VlmResult describeImage(File image, String modelName) throws Exception {
        String base64Image = Base64.getEncoder().encodeToString(Files.readAllBytes(image.toPath()));

        // request body: one user message with 2 parts, the image + the prompt
        JsonObject inlineData = new JsonObject();
        inlineData.addProperty("mime_type", mimeType(image));
        inlineData.addProperty("data", base64Image);
        JsonObject imagePart = new JsonObject();
        imagePart.add("inline_data", inlineData);

        JsonObject textPart = new JsonObject();
        textPart.addProperty("text", PROMPT);

        JsonArray parts = new JsonArray();
        parts.add(imagePart);
        parts.add(textPart);
        JsonObject content = new JsonObject();
        content.add("parts", parts);
        JsonArray contents = new JsonArray();
        contents.add(content);

        // forces the answer to be JSON instead of free text
        JsonObject generationConfig = new JsonObject();
        generationConfig.addProperty("responseMimeType", "application/json");

        JsonObject body = new JsonObject();
        body.add("contents", contents);
        body.add("generationConfig", generationConfig);

        String response = HttpUtil.postJson(GEMINI_URL + model + ":generateContent", gson.toJson(body), "x-goog-api-key", apiKey);
        String answer = extractText(response);

        VlmResult result = gson.fromJson(cleanJson(answer), VlmResult.class);
        result.image = image.getName();
        return result;
    }

    // Gemini's answer sits in candidates[0].content.parts[].text
    private String extractText(String response) {
        JsonObject root = JsonParser.parseString(response).getAsJsonObject();
        JsonArray candidates = root.getAsJsonArray("candidates");
        if (candidates == null || candidates.size() == 0) {
            throw new RuntimeException("Gemini returned no answer: " + response);
        }
        JsonObject content = candidates.get(0).getAsJsonObject().getAsJsonObject("content");
        if (content == null || !content.has("parts")) {
            throw new RuntimeException("Gemini answer has no text (blocked?): " + response);
        }
        StringBuilder text = new StringBuilder();
        for (JsonElement part : content.getAsJsonArray("parts")) {
            JsonObject partObject = part.getAsJsonObject();
            // skip thinking parts, only keep the actual answer
            if (partObject.has("thought") && partObject.get("thought").getAsBoolean()) {
                continue;
            }
            if (partObject.has("text")) {
                text.append(partObject.get("text").getAsString());
            }
        }
        return text.toString();
    }

    // just in case it wraps the JSON in ```json ... ``` anyway
    private String cleanJson(String text) {
        int start = text.indexOf("{");
        int end = text.lastIndexOf("}");
        if (start == -1 || end == -1) {
            throw new RuntimeException("Gemini did not return JSON: " + text);
        }
        return text.substring(start, end + 1);
    }

    private String mimeType(File image) {
        String name = image.getName().toLowerCase();
        if (name.endsWith(".png")) {
            return "image/png";
        }
        if (name.endsWith(".webp")) {
            return "image/webp";
        }
        return "image/jpeg";
    }
}
