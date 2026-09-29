package mdip.vlm;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.Reader;
import java.io.Writer;
import java.lang.reflect.Type;
import java.util.LinkedHashMap;
import java.util.Map;

// the json file where every image is mapped to its objects, what they're
// doing and their relationships, keyed by image file name:
// { "dog.jpg": { "objects": [...], "actions": [...], "relationships": [...] }, ... }
public class VlmJsonStore {

    private final File jsonFile;
    private final Gson gson = new GsonBuilder().setPrettyPrinting().create();

    public VlmJsonStore(File jsonFile) {
        this.jsonFile = jsonFile;
    }

    public Map<String, VlmResult> loadAll() throws Exception {
        if (!jsonFile.exists()) {
            return new LinkedHashMap<>();
        }
        Type type = new TypeToken<LinkedHashMap<String, VlmResult>>() {}.getType();
        try (Reader reader = new FileReader(jsonFile)) {
            Map<String, VlmResult> results = gson.fromJson(reader, type);
            return results == null ? new LinkedHashMap<>() : results;
        }
    }

    public VlmResult get(String imageName) throws Exception {
        return loadAll().get(imageName);
    }

    // adds the image or overwrites its old entry, keeps the other images
    public void save(VlmResult result) throws Exception {
        Map<String, VlmResult> results = loadAll();
        results.put(result.image, result);
        try (Writer writer = new FileWriter(jsonFile)) {
            gson.toJson(results, writer);
        }
    }
}
