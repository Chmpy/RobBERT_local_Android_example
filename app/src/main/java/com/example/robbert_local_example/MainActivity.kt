package com.example.robbert_local_example

import Utils
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat.startActivity
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : ComponentActivity() {
    private lateinit var tflite: Interpreter
    private lateinit var vocab: Map<String, Int>
    private val MAX_SEQUENCE_LENGTH = 128 // Match this with your model's expected input size
    private val TAG = "MAIN_ACTIVITY"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.d(TAG, "Loading TFLite model and vocabulary")
        // Load the TFLite model
        tflite = Interpreter(loadModelFile("model.tflite"))

        // Load the vocabulary
        vocab = Utils.loadVocabulary(this, "vocab.json")

        val attentionMaskTensor = tflite.getInputTensor(0)
        val inputTensor = tflite.getInputTensor(1)
        val outputTensor = tflite.getOutputTensor(0)
        Log.d(TAG, "Input and output tensor details:")
        Log.d(TAG, "Input tensor name: ${attentionMaskTensor.name()}")
        Log.d(TAG, "Input tensor index: ${attentionMaskTensor.index()}")
        Log.d(TAG, "Input tensor shape: ${attentionMaskTensor.shape().contentToString()}")
        Log.d(TAG, "Input tensor data type: ${attentionMaskTensor.dataType()}")
        Log.d(TAG, "Input tensor name: ${inputTensor.name()}")
        Log.d(TAG, "Input tensor index: ${inputTensor.index()}")
        Log.d(TAG, "Input tensor shape: ${inputTensor.shape().contentToString()}")
        Log.d(TAG, "Input tensor data type: ${inputTensor.dataType()}")
        Log.d(TAG, "Output tensor name: ${outputTensor.name()}")
        Log.d(TAG, "Output tensor index: ${outputTensor.index()}")
        Log.d(TAG, "Output tensor shape: ${outputTensor.shape().contentToString()}")
        Log.d(TAG, "Output tensor data type: ${outputTensor.dataType()}")

        setContent {
            MaterialTheme {
                MainScreen(::performInference, ::startSpeedTestService)
            }
        }
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
        // Prepare and tokenize input
        Log.d(TAG, "Starting inference for input: $input")
        val (processedInput, maskPosition) = Utils.preprocessInput(input)
        Log.d(TAG, "Preprocessed input: $processedInput, Mask position: $maskPosition")
        if (maskPosition == null) {
            return "No mask token found in input"
        }
        var inputIds = Utils.tokenize(processedInput, vocab)

        // Prepare input for inference
        inputIds = listOf(vocab["<s>"] ?: 0) + inputIds + listOf(vocab["</s>"] ?: 0)
        inputIds = Utils.padOrTruncate(inputIds, MAX_SEQUENCE_LENGTH, vocab["<pad>"] ?: 1)
        val attentionMask = Utils.createAttentionMask(inputIds.size)

        // Create TensorBuffers for input and output
        val inputBuffer = LongBuffer.wrap(inputIds.map { it.toLong() }.toLongArray())
        Log.d(TAG, "Input details: ${inputBuffer.array().contentToString()}")
        val maskBuffer = LongBuffer.wrap(attentionMask)
        Log.d(TAG, "Attention mask details: ${maskBuffer.array().contentToString()}")
        val outputBuffer = FloatBuffer.allocate(MAX_SEQUENCE_LENGTH * vocab.size)

        // Run inference and get top predictions
        try {
            Log.d(TAG, "Running TFLite inference")
            tflite.runForMultipleInputsOutputs(
                arrayOf(maskBuffer, inputBuffer),
                mapOf(0 to outputBuffer)
            )
            Log.d(TAG, "TFLite inference completed successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error during TFLite inference", e)
            return "Error during inference: ${e.message}"
        }

        var predictions = Utils.getTopPredictions(
            outputBuffer,
            vocab,
            maskPosition,
        )

        // Post-process predictions to remove tokenization artifacts
        predictions = predictions.map { it.replace("Ġ", "") }

        return predictions.joinToString(", ")
    }

    private fun startSpeedTestService() {
        val intent = Intent(this, SpeedTestService::class.java)
        startService(intent)
    }
}

@Composable
fun MainScreen(
    onInference: (String) -> String,
    startSpeedTest: () -> Unit
) {
    var input by remember { mutableStateOf("") }
    var result by remember { mutableStateOf("") }
    var newSentence by remember { mutableStateOf("") }

    // Function to show the corrected sentence
    fun getTopPrediction(input: String, prediction: String, maskPosition: Int): String {
        if (maskPosition == 0) return ""
        val words = input.split(" ")
        val updatedWords = words.toMutableList()
        updatedWords[maskPosition] = prediction.split(", ").first()
        return updatedWords.joinToString(" ")
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        TextField(
            value = input,
            onValueChange = { input = it },
            label = { Text("Enter a sentence") },
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = {
            result = onInference(input)
            val (_, maskPosition) = Utils.preprocessInput(input)
            newSentence = getTopPrediction(input, result, maskPosition ?: 0)
        }) {
            Text("Perform Inference")
        }

        Spacer(modifier = Modifier.height(16.dp))

        Column(
            modifier = Modifier.fillMaxWidth(),
            horizontalAlignment = Alignment.Start
        ) {
            Text("Top predictions: $result")
            Spacer(modifier = Modifier.height(16.dp))
            Text("New sentence: $newSentence")
        }

        Spacer(modifier = Modifier.weight(1f)) // Spacer with weight pushes the button down

        Button(onClick = {
            startSpeedTest()
        }) {
            Text("Speed Test")
        }
    }
}