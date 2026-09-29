package mdip.vit;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import mdip.VectorUtil;

public class ViTSearcher {

    public static class Result {
        public final String imagePath;
        public final double score;

        public Result(String imagePath, double score) {
            this.imagePath = imagePath;
            this.score = score;
        }
    }

    private final File indexFile;

    public ViTSearcher(File indexFile) {
        this.indexFile = indexFile;
    }

    public List<Result> search(double[] queryVector, int topK) throws Exception {
        List<Result> results = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(indexFile))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split("\t");
                if (parts.length != 2) {
                    continue;
                }
                double[] vector = VectorUtil.parseCsv(parts[1]);
                double score = VectorUtil.cosineSimilarity(queryVector, vector);
                results.add(new Result(parts[0], score));
            }
        }
        results.sort(Comparator.comparingDouble((Result r) -> r.score).reversed());
        return results.subList(0, Math.min(topK, results.size()));
    }
}
