package com.example.robbert_local_example

import android.app.Service
import android.content.Intent
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import android.widget.Toast
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.system.measureTimeMillis

class SpeedTestService : Service() {

    private lateinit var tflite: Interpreter
    private lateinit var vocab: Map<String, Int>
    private val MAX_SEQUENCE_LENGTH = 128
    private val TAG = "SPEED_TEST_SERVICE"

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "Speed test service started")

        // Load model and vocabulary
        tflite = Interpreter(loadModelFile("model.tflite"))
        vocab = Utils.loadVocabulary(this, "vocab.json")

        // Run the speed test in a separate thread
        Thread {
            runSpeedTest()
            stopSelf()  // Stop the service after completing the task
        }.start()

        return START_STICKY
    }

    private fun runSpeedTest() {
        val sentences = listOf(
            "Ik heb een vriend die altijd te laat komt.", // Correct
            "Ik weet die ik het kan.", // Incorrect
            "Ik weet dat ik het kan.", // Correct
            "Daarom is het belangrijk, je moet goed opletten.", // Incorrect
            "Ik ken een man die altijd grapjes maakt.", // Correct
            "Ze heeft een jurk gekocht die perfect past.", // Correct
            "Er is een boek dat ik je echt kan aanraden.", // Correct
            "We bezochten een stad die bekend staat om haar architectuur.", // Correct
            "Hij las een artikel dat zijn mening veranderde.", // Correct
            "Ze vertelde over een ervaring die haar leven veranderde.", // Correct
            "Ik zag een film die mij aan het denken zette.", // Correct
            "Hij gebruikt een methode die zeer effectief is.", // Correct
            "Het kind dat in de tuin speelt, is mijn neefje.", // Correct
            "De vraag die ze stelde, was erg moeilijk.", // Correct
            "De beslissing dat hij maakte, was erg moeilijk.", // Incorrect
            "Dit is de laptop die ik wil kopen.", // Correct
            "Dit is de laptop dat ik wil kopen.", // Incorrect
            "Het project dat we gestart zijn, verloopt goed.", // Correct
            "De vrouw die je daar ziet, is mijn lerares.", // Correct
            "De resultaten dat we behaalden, waren indrukwekkend.", // Incorrect
            "De resultaten die we behaalden, waren indrukwekkend.", // Correct
            "De bloemen die je hebt meegenomen, zijn prachtig.", // Correct
            "De informatie dat hij gaf, was nuttig.", // Incorrect
            "De informatie die hij gaf, was nuttig.", // Correct
            "Het idee dat ze voorstelde, was briljant.", // Correct
            "Het idee die ze voorstelde, was briljant.", // Incorrect
            "De kat dat op het dak zit, is van ons.", // Incorrect
            "De kat die op het dak zit, is van ons.", // Correct
            "De film die we gisteren keken, was spannend.", // Correct
            "De auto dat ik gisteren zag, was heel mooi.", // Incorrect
        )

        // Use a Handler to post Toast messages to the main thread
        val mainHandler = Handler(Looper.getMainLooper())

        mainHandler.post {
            Toast.makeText(this, "Starting speed test", Toast.LENGTH_SHORT).show()
        }

        val totalTime = measureExecutionTime {
            for (sentence in sentences) {
                Log.d(TAG, "Processing sentence: $sentence")
                performInference(sentence)
                Log.d(TAG, "")
            }
        }

        Log.d(TAG, "Total time taken: $totalTime s")
        Log.d(TAG, "Average time per sentence: ${"%.3f".format(totalTime / sentences.size)} s")

        mainHandler.post {
            Toast.makeText(this, "Total time taken: $totalTime s", Toast.LENGTH_SHORT).show()
            Toast.makeText(this, "Average time per sentence: ${"%.3f".format(totalTime / sentences.size)} s", Toast.LENGTH_SHORT).show()
        }
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

    private inline fun measureExecutionTime(block: () -> Unit): Double {
        val elapsedTimeMillis = measureTimeMillis {
            block()
        }
        return elapsedTimeMillis / 1000.0 // Convert milliseconds to seconds
    }

    override fun onBind(intent: Intent?): IBinder? {
        return null  // This is a started service, not a bound service
    }
}