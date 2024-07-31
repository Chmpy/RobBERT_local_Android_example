import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.nio.FloatBuffer
import kotlin.math.exp

object Utils {
    private const val TAG = "UTILS"

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

    fun loadVocabulary(context: Context, filename: String): Map<String, Int> {
        Log.d(TAG, "Loading vocabulary from $filename")
        val json = context.assets.open(filename).bufferedReader().use { it.readText() }
        val vocab: Map<String, Int> =
            Gson().fromJson(json, object : TypeToken<Map<String, Int>>() {}.type)
        Log.d(TAG, "Vocabulary loaded. Size: ${vocab.size}")
        return vocab
    }

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

    fun createAttentionMask(tokenCount: Int): LongArray {
        Log.d(TAG, "Creating attention mask of length $tokenCount")
        return LongArray(tokenCount) { 1L }
    }

    fun padOrTruncate(ids: List<Int>, maxLength: Int, padToken: Int): List<Int> {
        val result = if (ids.size > maxLength) {
            ids.take(maxLength)
        } else {
            ids + List(maxLength - ids.size) { padToken }
        }
        Log.d(TAG, "Padding/truncating complete. New length: ${result.size}")
        return result
    }

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

        //Divide the logits into vocab.size chunks
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

    private fun topKIndices(x: FloatArray, k: Int): IntArray {
        return x.withIndex()
            .sortedByDescending { it.value }
            .take(k)
            .map { it.index }
            .toIntArray()
    }

    private fun softmax(input: FloatArray): FloatArray {
        val max = input.maxOrNull() ?: 0f
        val exp = input.map { exp((it - max).toDouble()).toFloat() }
        val sum = exp.sum()
        return exp.map { it / sum }.toFloatArray()
    }
}