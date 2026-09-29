package mdip;

import java.io.File;

// the 8 test images, label shown in the menu + file name inside images/
// drop your images in with these exact names (or change the names here)
public class ImageCatalog {

    public static final String IMAGES_DIR = "images";

    public static final String[] LABELS = {
            "dog image (on grass)",
            "cat image (on grass)",
            "handshake image 1",
            "handshake image 2",
            "handshake image 3",
            "teamwork image",
            "whale image",
            "car image",
            "group of people image",
    };

    public static final String[] FILES = {
            "dog.jpeg",
            "cat.jpeg",
            "handshake.jpg",
            "handshake-2.jpg",
            "handshake-3.jpeg",
            "teamwork.jpg",
            "whale.jpg",
            "car.jpeg",
            "group.jpeg",
    };

    public static int count() {
        return FILES.length;
    }

    public static File getFile(int index) {
        return new File(IMAGES_DIR, FILES[index]);
    }

    // index files store the full path, this turns it back into "dog image (on grass)"
    public static String labelFor(String path) {
        String name = new File(path).getName();
        for (int i = 0; i < FILES.length; i++) {
            if (FILES[i].equals(name)) {
                return LABELS[i];
            }
        }
        return name;
    }
}
