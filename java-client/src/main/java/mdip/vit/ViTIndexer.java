package mdip.vit;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import mdip.VectorUtil;

public class ViTIndexer {

    private final ViTClient client;
    private final File indexFile;
    private static final String[] IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"};


    public ViTIndexer(ViTClient client, File indexFile) {
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

    public void close() {
    }
}
