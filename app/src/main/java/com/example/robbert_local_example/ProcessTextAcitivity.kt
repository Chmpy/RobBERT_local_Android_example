package com.example.robbert_local_example

import Utils
import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ProcessTextActivity : Activity() {

    private lateinit var tflite: Interpreter
    private lateinit var vocab: Map<String, Int>
    private val MAX_SEQUENCE_LENGTH = 128
    private val TAG = "PROCESS_TEXT_ACTIVITY"
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Check if the intent is for processing text and is not read-only
        if (intent?.action == Intent.ACTION_PROCESS_TEXT &&
            intent.type == "text/plain" &&
            !intent.getBooleanExtra("readonly", false)
        ) {
            val text = intent.getStringExtra(Intent.EXTRA_PROCESS_TEXT)

            if (text == null) {
                Toast.makeText(this, "No text to process", Toast.LENGTH_SHORT).show()
                finish()
                return
            }

            Log.d(TAG, "Text to process: $text")
            Log.d(TAG, "Loading TFLite model and vocabulary")
            tflite = Interpreter(loadModelFile("model.tflite"))
            vocab = Utils.loadVocabulary(this, "vocab.json")

            handleSelectedText(text)

        } else {
            Toast.makeText(this, "Text cannot be modified", Toast.LENGTH_SHORT).show()
        }
        finish()
    }

    private fun handleSelectedText(selectedText: String) {

        Log.d(TAG, "Selected text: $selectedText")

        // Preprocess the input text and get the mask position
        val (_, maskPosition) = Utils.preprocessInput(selectedText)

        if (maskPosition == null) {
            Toast.makeText(this, "No mask token found in input", Toast.LENGTH_SHORT).show()
            return
        }

        // Perform inference and get the top prediction, then replace the mask token with the prediction
        val feedback = performInference(selectedText).split(", ").first()
        val feedbackText = selectedText.split(" ").toMutableList().apply {
            set(maskPosition, feedback)
        }.joinToString(" ")

        // Set the result with the processed text
        val intent = Intent(Intent.EXTRA_PROCESS_TEXT)
        intent.putExtra(Intent.EXTRA_PROCESS_TEXT, feedbackText)
        setResult(RESULT_OK, intent)
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        Log.d(TAG, "Loading model file: $modelName")
        val fileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        Log.d(TAG, "Model file loaded successfully")
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun performInference(input: String): String {
        Log.d(TAG, "Starting inference for input: $input")
        val (processedInput, maskPosition) = Utils.preprocessInput(input)
        Log.d(TAG, "Preprocessed input: $processedInput, Mask position: $maskPosition")

        if (maskPosition == null) {
            return "No mask token found in input"
        }

        var inputIds = Utils.tokenize(processedInput, vocab)

        // Add start and end tokens
        inputIds = listOf(vocab["<s>"] ?: 0) + inputIds + listOf(vocab["</s>"] ?: 0)

        // Pad or truncate to MAX_SEQUENCE_LENGTH
        inputIds = Utils.padOrTruncate(inputIds, MAX_SEQUENCE_LENGTH, vocab["<pad>"] ?: 1)

        // Create attention mask
        val attentionMask = Utils.createAttentionMask(inputIds.size)

        // Convert input to LongBuffer
        val inputBuffer = LongBuffer.wrap(inputIds.map { it.toLong() }.toLongArray())
        Log.d(TAG, "Input details: ${inputBuffer.array().contentToString()}")

        // Convert attention mask to LongBuffer
        val maskBuffer = LongBuffer.wrap(attentionMask)
        Log.d(TAG, "Attention mask details: ${maskBuffer.array().contentToString()}")

        val outputBuffer = FloatBuffer.allocate(MAX_SEQUENCE_LENGTH * vocab.size)

        try {
            Log.d(TAG, "Running TFLite inference")
            tflite.runForMultipleInputsOutputs(
                arrayOf(maskBuffer, inputBuffer), mapOf(0 to outputBuffer)
            )
            Log.d(TAG, "TFLite inference completed successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error during TFLite inference", e)
            return "Error during inference: ${e.message}"
        }

        val predictions = Utils.getTopPredictions(
            outputBuffer, vocab, maskPosition, topK = 3
        )

        return predictions.joinToString(", ")
    }
}