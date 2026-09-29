package mdip.vlm;

import java.util.ArrayList;
import java.util.List;

// same shape as the JSON Gemini sends back and as vlm-results.json:
// objects -> what each one is doing (actions) -> how they sit relative
// to each other (relationships), which is the Srel / Sprel part of the course
public class VlmResult {

    public String image;
    public List<String> objects = new ArrayList<>();
    public List<Action> actions = new ArrayList<>();
    public List<Relationship> relationships = new ArrayList<>();
    public String description = "";

    // {"object": "person 1", "action": "shaking hands with person 2"}
    public static class Action {
        public String object;
        public String action;
    }

    // {"subject": "dog", "relation": "on top of", "object": "grass"}
    public static class Relationship {
        public String subject;
        public String relation;
        public String object;
    }
}
