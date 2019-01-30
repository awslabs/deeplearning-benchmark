package mxnet

import org.apache.mxnet.infer.{ImageClassifier, ObjectDetector}
import org.apache.mxnet._
import org.kohsuke.args4j.{CmdLineParser, Option}

import scala.collection.JavaConverters._

object SSDObjectDetection {

  class CLIParser {

    @Option(name = "--modelPathPrefix", usage = "The model to benchmark")
    val modelPathPrefix: String = "/tmp/resnet50_ssd/resnet50_ssd_model"

    @Option(name = "--inputImagePath", usage = "Input image path for single inference")
    val inputImagePath: String = "/tmp/resnet50_ssd/images/dog.jpg"

    @Option(name = "--batchSize", usage = "Batchsize of the model")
    val batchSize: Int = 1

    @Option(name = "--times", usage = "Number of times to run the benchmark")
    val times: Int = 1

    @Option(name = "--context", usage = "Context to run on")
    val context: String = "cpu"

  }

  def loadModel(modelPathPrefix: String, context: Array[Context], batchSize: Int): ObjectDetector = {
    val dType = DType.Float32
    val inputShape = Shape(batchSize, 3, 512, 512)
    val inputDescriptor = IndexedSeq(DataDesc("data", inputShape, dType, "NCHW"))

    new ObjectDetector(modelPathPrefix, inputDescriptor, context)
  }

  def loadSingleImage(inputImagePath: String, context: Array[Context]): NDArray = {
    val img = ImageClassifier.loadImageFromFile(inputImagePath)
    val reShapedImg = ImageClassifier.reshapeImage(img, 512, 512)
    val imgND = ImageClassifier.bufferedImageToPixels(reShapedImg, Shape(1, 3, 512, 512))

    imgND.copyTo(context(0))
  }

  def loadBatchImage(inputImagePath: String, batchSize: Int, context: Array[Context]): NDArray = {

    val imgND = loadSingleImage(inputImagePath, context)
    val listND = List.fill(batchSize) (imgND.copy()).toArray

    val concatNDArray = NDArray.api.concat(listND, batchSize, Some(0))

    println(concatNDArray.shape)

    concatNDArray.copyTo(context(0))

  }

  def runSingleInference(modelPathPrefix: String, inputImagePath: String, context: Array[Context], batchSize: Int = 1, times: Int): List[Double] = {
    val loadedModel = loadModel(modelPathPrefix, context, batchSize)

    val dataSet = loadSingleImage(inputImagePath, context)

    var inferenceTimes : List[Double] = List()

    // Warm up intervals
    println("Warming up the system")
    for (i <- 1 to 5) {
      NDArrayCollector.auto().withScope {
        val output = loadedModel.objectDetectWithNDArray(IndexedSeq(dataSet), Some(3))
      }
    }
    println("Warm up done")

    for (i <- 1 to times) {
      NDArrayCollector.auto().withScope {

        val startTime = System.nanoTime()
        val output = loadedModel.objectDetectWithNDArray(IndexedSeq(dataSet), Some(3))
        val estimatedTime = (System.nanoTime() - startTime) / (1e6 * 1.0)
        inferenceTimes = estimatedTime :: inferenceTimes

        println("Inference time at iteration: %d is : %f \n".format(i, estimatedTime))
      }
    }

    inferenceTimes
  }

  def runBatchInference(modelPathPrefix: String, inputImagePath: String, context: Array[Context], batchSize: Int, times: Int): List[Double] = {

    val loadedModel = loadModel(modelPathPrefix, context, batchSize)

    val dataSet = loadBatchImage(inputImagePath, batchSize, context)

    var inferenceTimes : List[Double] = List()

    // Warm up intervals
    println("Warming up the system")
    for (i <- 1 to 5) {
      NDArrayCollector.auto().withScope {
        val output = loadedModel.objectDetectWithNDArray(IndexedSeq(dataSet), Some(3))
      }
    }
    println("Warm up done")

    for (i <- 1 to times) {
      NDArrayCollector.auto().withScope {

        val startTime = System.nanoTime()
        val output = loadedModel.objectDetectWithNDArray(IndexedSeq(dataSet), Some(3))
        val estimatedTime = (System.nanoTime() - startTime) / (1e6 * 1.0)
        inferenceTimes = estimatedTime :: inferenceTimes

        println("Inference time at iteration: %d is : %f \n".format(i, estimatedTime))
      }

    }

    inferenceTimes

  }


  def main(args: Array[String]): Unit = {

    val inst = new CLIParser

    val parser: CmdLineParser = new CmdLineParser(inst)

    parser.parseArgument(args.toList.asJava)

    val context = Utils.getContext(inst.context)

    val modelPathPrefix = inst.modelPathPrefix

    val inputImagePath = inst.inputImagePath

    val batchSize = inst.batchSize

    val times = inst.times


    NDArrayCollector.auto().withScope {
      println("Running single inference")
      val inferenceTimeSingle = runSingleInference(modelPathPrefix, inputImagePath, context, 1, times)

      Utils.printStatistics(inferenceTimeSingle, "single_inference")
    }

    NDArrayCollector.auto().withScope {
      println("Running batch inference with batsize : " + (batchSize))
      val inferenceTimeBatch = runBatchInference(modelPathPrefix, inputImagePath, context, batchSize, times)

      Utils.printStatistics(inferenceTimeBatch, "batch_inference_1x")

    }

    NDArrayCollector.auto().withScope {
      println("Running batch inference with batsize : " + (batchSize * 2))
      val inferenceTimeBatch = runBatchInference(modelPathPrefix, inputImagePath, context, batchSize * 2, times)

      Utils.printStatistics(inferenceTimeBatch, "batch_inference_2x")

    }

    NDArrayCollector.auto().withScope {
      println("Running batch inference with batsize : " + (batchSize * 4))
      val inferenceTimeBatch = runBatchInference(modelPathPrefix, inputImagePath, context, batchSize * 4, times)

      Utils.printStatistics(inferenceTimeBatch, "batch_inference_4x")

    }
  }

}
