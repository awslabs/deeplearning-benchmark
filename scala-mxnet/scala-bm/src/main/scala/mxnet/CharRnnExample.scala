package mxnet

import org.apache.mxnet.{Context, Model, NDArrayCollector}
import org.kohsuke.args4j.{CmdLineParser, Option}

import scala.collection.JavaConverters._

object CharRnnExample {

  private var vocab : Map[String, Int] = null

  class CLIParser {

    @Option(name = "--modelPathPrefix", usage = "the model prefix")
    val modelPathPrefix: String = "./model/obama"

    @Option(name = "--data-path", usage = "the input train data file")
    val dataPath: String = "./data/obama.txt"

    @Option(name = "--starter-sentence", usage = "the starter sentence")
    val starterSentence: String = "The joke"

    @Option(name = "--times", usage = "Number of times to run the benchmark")
    val times: Int = 1

    @Option(name = "--context", usage = "Context to run on")
    val context: String = "cpu"

  }


  def loadModel(modelPathPrefix: String, dataPath: String, context: Context): RnnModel.LSTMInferenceModel = {

    val buckets = List(129)
    val numHidden = 512
    val numEmbed = 256
    val numLstmLayer = 3

    val (_, argParams, _) = Model.loadCheckpoint(modelPathPrefix, 75)
    this.vocab = Utils.buildVocab(dataPath)

    val model = new RnnModel.LSTMInferenceModel(numLstmLayer, vocab.size + 1, numHidden = numHidden, numEmbed = numEmbed, numLabel = vocab.size + 1, argParams = argParams, dropout = 0.2f, ctx = context)

    model
  }

  def runSingleInference(modelPathPrefix: String, dataPath: String, starterSentence: String, context: Context, times: Int): List[Double] = {

    val model = loadModel(modelPathPrefix, dataPath, context)

    var inferenceTimes : List[Double] = List()

    // Warm up intervals
    println("Warming up the system")
    for (i <- 1 to 5) {
      NDArrayCollector.auto().withScope {
        val output = Utils.runPrediction(model, starterSentence, vocab)
      }
    }
    println("Warm up done")

    for (i <- 1 to times) {
      NDArrayCollector.auto().withScope {

        val startTime = System.nanoTime()
        val output = Utils.runPrediction(model, starterSentence, vocab)
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

    var context = Utils.getContext(inst.context)

    val modelPathPrefix = inst.modelPathPrefix

    val dataPath = inst.dataPath

    val starterSentence = inst.starterSentence

    val times = inst.times

    NDArrayCollector.auto().withScope {
      println("Running single inference")
      val inferenceTimeSingle = runSingleInference(modelPathPrefix, dataPath, starterSentence, context, times)

      Utils.printStatistics(inferenceTimeSingle, "single_inference")

    }

  }

}
