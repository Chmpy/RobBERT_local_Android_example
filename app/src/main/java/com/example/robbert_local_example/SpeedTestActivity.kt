package com.example.robbert_local_example

import Utils
import android.app.Activity
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.system.measureTimeMillis

class SpeedTestActivity : Activity() {

    private lateinit var tflite: Interpreter
    private lateinit var vocab: Map<String, Int>
    private val MAX_SEQUENCE_LENGTH = 128
    private val TAG = "SPEED_TEST_ACTIVITY"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        tflite = Interpreter(loadModelFile("model.tflite"))
        vocab = Utils.loadVocabulary(this, "vocab.json")

        val sentences = listOf(
            "Ik heb een vriend die altijd te laat komt.",
            "Ik weet die ik het kan.",
            "Ik weet DAT ik het kan.",
            "Daarom is het belangrijk, je moet goed opletten.",
            "Ik ken een man die altijd grapjes maakt.",
            "Ze heeft een jurk gekocht die perfect past.",
            "Er is een boek dat ik je echt kan aanraden.",
            "We bezochten een stad die bekend staat om haar architectuur.",
            "Hij las een artikel dat zijn mening veranderde.",
            "Ze vertelde over een ervaring die haar leven veranderde.",
            "Ik zag een film die mij aan het denken zette.",
            "Hij gebruikt een methode die zeer effectief is.",
        )

        Log.d(TAG, "Starting speed test")
        Toast.makeText(this, "Starting speed test", Toast.LENGTH_SHORT).show()
        val totalTime = measureExecutionTime {
            for (sentence in sentences) {
                Log.d(TAG, "Processing sentence: $sentence")
                performInference(sentence)
                Log.d(TAG, "")
            }
        }
        Log.d(TAG, "Total time taken: $totalTime s")
        Toast.makeText(this, "Total time taken: $totalTime s", Toast.LENGTH_SHORT).show()
        finish()
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun performInference(input: String) {
        val (processedInput, maskPosition) = Utils.preprocessInput(input)

        if (maskPosition == null) {
            return
        }

        var inputIds = Utils.tokenize(processedInput, vocab)
        inputIds = listOf(vocab["<s>"] ?: 0) + inputIds + listOf(vocab["</s>"] ?: 0)
        inputIds = Utils.padOrTruncate(inputIds, MAX_SEQUENCE_LENGTH, vocab["<pad>"] ?: 1)
        val attentionMask = Utils.createAttentionMask(inputIds.size)
        val inputBuffer = LongBuffer.wrap(inputIds.map { it.toLong() }.toLongArray())
        val maskBuffer = LongBuffer.wrap(attentionMask)
        val outputBuffer = FloatBuffer.allocate(MAX_SEQUENCE_LENGTH * vocab.size)

        val inferenceTime = measureExecutionTime {
            tflite.runForMultipleInputsOutputs(
                arrayOf(maskBuffer, inputBuffer), mapOf(0 to outputBuffer)
            )
        }
        Log.d(TAG, "Inference time taken: $inferenceTime s")
    }
}

private inline fun measureExecutionTime(block: () -> Unit): Double {
    val elapsedTimeMillis = measureTimeMillis {
        block()
    }
    return elapsedTimeMillis / 1000.0 // Convert milliseconds to seconds
}
