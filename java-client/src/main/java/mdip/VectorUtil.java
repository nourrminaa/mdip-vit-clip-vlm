package mdip;

public class VectorUtil {

    public static double[] parseCsv(String csv) {
        String[] parts = csv.trim().split(",");
        double[] vector = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            vector[i] = Double.parseDouble(parts[i].trim());
        }
        return vector;
    }

    public static String toCsv(double[] vector) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < vector.length; i++) {
            if (i > 0) {
                builder.append(",");
            }
            builder.append(vector[i]);
        }
        return builder.toString();
    }

    public static double cosineSimilarity(double[] a, double[] b) {
        double dot = 0;
        double normA = 0;
        double normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        if (normA == 0 || normB == 0) {
            return 0;
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    // turns cosine scores into probabilities that add up to 1 (softmax).
    // CLIP multiplies the cosine by its learned temperature (logit scale = 100)
    // before the softmax, without it every probability comes out almost equal
    // since cosines between CLIP vectors are all squeezed around 0.2-0.35
    public static double[] softmax(double[] scores, double scale) {
        double max = Double.NEGATIVE_INFINITY;
        for (double score : scores) {
            max = Math.max(max, score * scale);
        }
        double[] result = new double[scores.length];
        double sum = 0;
        for (int i = 0; i < scores.length; i++) {
            // subtracting the max changes nothing mathematically, it just
            // stops exp() from overflowing
            result[i] = Math.exp(scores[i] * scale - max);
            sum += result[i];
        }
        for (int i = 0; i < result.length; i++) {
            result[i] = result[i] / sum;
        }
        return result;
    }

    // first few values of a feature vector, printing all 768 is useless
    public static String preview(double[] vector, int count) {
        StringBuilder builder = new StringBuilder("[");
        for (int i = 0; i < Math.min(count, vector.length); i++) {
            if (i > 0) {
                builder.append(", ");
            }
            builder.append(String.format("%.4f", vector[i]));
        }
        builder.append(", ...] (").append(vector.length).append(" values)");
        return builder.toString();
    }
}
