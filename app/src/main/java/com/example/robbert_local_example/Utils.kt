import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.nio.FloatBuffer
import kotlin.math.exp

object Utils {
    private const val TAG = "UTILS"

    /**
     * Mask "die" or "dat" in the input sentence.
     *
     * @param sentence The input sentence.
     * @return A pair containing the preprocessed sentence and the position of the mask, or null if no mask was applied.
     */
    fun preprocessInput(sentence: String): Pair<String, Int?> {
        Log.d(TAG, "Preprocessing input: $sentence")
        val words = sentence.split(" ")
        var maskPosition: Int? = null
        val processedWords = words.mapIndexed { index, word ->
            if (word.lowercase() in listOf("die", "dat") && maskPosition == null) {
                maskPosition = index
                "<mask>"
            } else {
                word
            }
        }
        val result = Pair(processedWords.joinToString(" "), maskPosition)
        Log.d(TAG, "Preprocessed result: $result")
        return result
    }

    /**
     * Load vocabulary from a JSON file located in the assets folder.
     *
     * @param context The context from which to access the assets.
     * @param filename The name of the JSON file containing the vocabulary.
     * @return A map representing the vocabulary, where keys are words and values are their corresponding indices.
     */
    fun loadVocabulary(context: Context, filename: String): Map<String, Int> {
        Log.d(TAG, "Loading vocabulary from $filename")
        val json = context.assets.open(filename).bufferedReader().use { it.readText() }
        val vocab: Map<String, Int> =
            Gson().fromJson(json, object : TypeToken<Map<String, Int>>() {}.type)
        Log.d(TAG, "Vocabulary loaded. Size: ${vocab.size}")
        return vocab
    }

    /**
     * Tokenize a sentence using the provided vocabulary.
     *
     * @param sentence The input sentence to tokenize.
     * @param vocab The vocabulary map to use for tokenization.
     * @return A list of integers representing the tokenized sentence.
     */
    fun tokenize(sentence: String, vocab: Map<String, Int>): List<Int> {
        Log.d(TAG, "Tokenizing sentence: $sentence")
        val maskTokenId = vocab["<mask>"] ?: vocab["<unk>"] ?: 0
        val pattern = Regex("<mask>|\\w+|[^\\w\\s]")
        val splitSentence = pattern.findAll(sentence).map { it.value }.toList()
        Log.d(TAG, "Split sentence: $splitSentence")
        val tokenizedSentence = mutableListOf<Int>()
        for (token in splitSentence) {
            when {
                token == "<mask>" -> tokenizedSentence.add(maskTokenId)
                token.matches(Regex("[^\\w\\s]")) -> {
                    (vocab["Ġ"] ?: vocab["<unk>"])?.let { tokenizedSentence.add(it) }
                    (vocab[token] ?: vocab["<unk>"])?.let { tokenizedSentence.add(it) }
                }

                else -> {
                    (vocab["Ġ$token"] ?: vocab["<unk>"])?.let { tokenizedSentence.add(it) }
                }
            }
        }

        Log.d(TAG, "Tokenized sentence: $tokenizedSentence")
        return tokenizedSentence
    }

    /**
     * Create an attention mask for a given number of tokens.
     *
     * @param tokenCount The number of tokens for which to create the attention mask.
     * @return A long array representing the attention mask, with all values set to 1.
     */
    fun createAttentionMask(tokenCount: Int): LongArray {
        Log.d(TAG, "Creating attention mask of length $tokenCount")
        return LongArray(tokenCount) { 1L }
    }

    /**
     * Pad or truncate a list of token IDs to a specified maximum length.
     *
     * @param ids The list of token IDs to pad or truncate.
     * @param maxLength The maximum length of the resulting list.
     * @param padToken The token ID to use for padding.
     * @return A list of token IDs padded or truncated to the specified length.
     */
    fun padOrTruncate(ids: List<Int>, maxLength: Int, padToken: Int): List<Int> {
        val result = if (ids.size > maxLength) {
            ids.take(maxLength)
        } else {
            ids + List(maxLength - ids.size) { padToken }
        }
        Log.d(TAG, "Padding/truncating complete. New length: ${result.size}")
        return result
    }

    /**
     * Get top-k predictions for the masked token.
     *
     * @param logits The model output logits as a FloatBuffer.
     * @param vocab The vocabulary map.
     * @param maskPosition The position of the mask token, or null if no mask is present.
     * @param topK The number of top predictions to return.
     * @return A list of the top-k predicted words.
     */
    fun getTopPredictions(
        logits: FloatBuffer,
        vocab: Map<String, Int>,
        maskPosition: Int? = null,
        topK: Int = 5
    ): List<String> {

        Log.d(TAG, "Getting top predictions for mask position: $maskPosition")

        if (maskPosition == null) {
            Log.e(TAG, "Mask position is null. Cannot get predictions.")
            return emptyList()
        }

        Log.d(TAG, "Logits shape: ${logits.capacity()}")

        // Divide the logits into vocab.size chunks
        val dividedLogits = logits.array().toList().chunked(vocab.size)

        // Get logits for the mask token
        val maskTokenLogits = dividedLogits[maskPosition + 1].toFloatArray()
        Log.d(TAG, "Logits at mask position: ${maskTokenLogits.take(10).toFloatArray().contentToString()}...")
        Log.d(TAG, "Logits at mask position size: ${maskTokenLogits.size}")

        // Apply softmax
        val probs = softmax(maskTokenLogits)
        Log.d(TAG, "Probabilities after softmax: ${probs.take(10).toFloatArray().contentToString()}...")

        // Get top-k indices
        val topIndices = topKIndices(probs, topK)
        Log.d(TAG, "Top $topK indices: ${topIndices.contentToString()}")

        // Convert indices to words
        val predictions = topIndices.map { index ->
            vocab.entries.find { it.value == index }?.key ?: "<unk>"
        }

        Log.d(TAG, "Top $topK predictions: $predictions")
        return predictions
    }

    /**
     * Get the top-k indices of a float array.
     *
     * @param x The float array.
     * @param k The number of top indices to retrieve.
     * @return An array of the top-k indices.
     */
    private fun topKIndices(x: FloatArray, k: Int): IntArray {
        return x.withIndex()
            .sortedByDescending { it.value }
            .take(k)
            .map { it.index }
            .toIntArray()
    }

    /**
     * Apply softmax to a float array.
     *
     * @param input The float array.
     * @return A float array representing the probabilities after applying softmax.
     */
    private fun softmax(input: FloatArray): FloatArray {
        val max = input.maxOrNull() ?: 0f
        val exp = input.map { exp((it - max).toDouble()).toFloat() }
        val sum = exp.sum()
        return exp.map { it / sum }.toFloatArray()
    }
}