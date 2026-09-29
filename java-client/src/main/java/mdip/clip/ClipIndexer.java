package mdip.clip;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import mdip.VectorUtil;

public class ClipIndexer {

    private static final String[] IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"};

    private final ClipClient client;
    private final File indexFile;

    public ClipIndexer(ClipClient client, File indexFile) {
        this.client = client;
        this.indexFile = indexFile;
    }

    public void createIndex(File imagesFolder) throws Exception {
        File[] images = imagesFolder.listFiles();
        if (images == null) {
            return;
        }
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(indexFile))) {
            for (File image : images) {
                if (!isImageFile(image)) {
                    continue;
                }
                double[] vector = client.embedImage(image);
                writer.write(image.getAbsolutePath());
                writer.write("\t");
                writer.write(VectorUtil.toCsv(vector));
                writer.newLine();
            }
        }
    }

    private boolean isImageFile(File file) {
        String name = file.getName().toLowerCase();
        for (String extension : IMAGE_EXTENSIONS) {
            if (name.endsWith(extension)) {
                return true;
            }
        }
        return false;
    }
}