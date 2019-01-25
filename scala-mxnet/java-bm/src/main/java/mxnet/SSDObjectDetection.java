package mxnet;

import org.apache.mxnet.infer.javaapi.*;
import org.apache.mxnet.javaapi.*;
import org.apache.mxnet.ResourceScope;
import org.kohsuke.args4j.Option;

import java.awt.image.BufferedImage
import org.kohsuke.args4j.CmdLineParser;

import java.util.ArrayList;
import java.util.List;

public class SSDObjectDetection {
    @Option(name = "--modelPathPrefix", usage = "The model to benchmark")
    String modelPathPrefix = "/tmp/resnet50_ssd/resnet50_ssd_model";
    @Option(name = "--inputImagePath", usage = "Input image path for single inference")
    String inputImagePath = "/tmp/resnet50_ssd/images/dog.jpg";
    @Option(name = "--batchSize", usage = "Batchsize of the model")
    int batchSize = 1;
    @Option(name = "--times", usage = "Number of times to run the benchmark")
    int times = 1;
    
    ObjectDetector loadModel(String modelPathPrefix, List<Context> context, int batchSize) {
        Shape inputShape = new Shape(new int[] {batchSize, 3, 512, 512});
        List<DataDesc> inputDescriptors = new ArrayList<>();
        inputDescriptors.add(new DataDesc("data", inputShape, DType.Float32(), "NCHW"));f
        return new ObjectDetector(modelPathPrefix, inputDescriptors, context, 0);
    }
    
    NDArray loadSingleImage(String inputImagePath, List<Context> context){
        BufferedImage img = ObjectDetector.loadImageFromFile(inputImagePath);
        BufferedImage reShapedImg = ObjectDetector.reshapeImage(img, 512, 512);
        NDArray imgND = ObjectDetector.bufferedImageToPixels(reShapedImg, new Shape(new int[] {1, 3, 512, 512}));
        return imgND.copyTo(context.get(0));
    }
    
    List<NDArray> loadBatchImage(String inputImagePath, int batchSize, List<Context> context) {
        NDArray imgND = loadSingleImage(inputImagePath, context);
        List<NDArray> nd = new ArrayList<>();
        NDArray[] temp = new NDArray[batchSize];
        for (int i = 0; i < batchSize; i++) temp[i] = imgND.copy();
        NDArray batched = NDArray.concat(temp, batchSize, 0, null)[0];
        nd.add(batched);
        return nd;
    }
    
    double[] runSingleInference(String modelPathPrefix, String inputImagePath, List<Context> context, int batchSize, int times) {
        ObjectDetector loadedModel = loadModel(modelPathPrefix, context, batchSize);
        List<double> inferenceTimes = new ArrayList();
        List<NDArray> dataset = new ArrayList<>();
        dataset.add(loadSingleImage(inputImagePath, context));
        
        // Warm up intervals
        // println("Warming up the system")
        for (int i = 0; i < 5; i++) {
            try(new ResourceScope()) {
                List<List<ObjectDetectorOutput>> output = loadedModel.objectDetectWithNDArray(dataset, 3);
            }
        }
        // println("Warm up done")
        
        double[] result = new double[times];
        for (int i = 0; i < times; i++) {
            try(new ResourceScope()) {
                long startTime = System.nanoTime();
                List<List<ObjectDetectorOutput>> output = loadedModel.objectDetectWithNDArray(dataset, 3);
                result[i] = (System.nanoTime() - startTime) / (1e6 * 1.0);
                
                
                // println("Inference time at iteration: %d is : %f \n".format(i, estimatedTime))
            }
        }
        return result;
    }
    
    public static void main(String[] args) {
        CmdLineParser parser = new CmdLineParser();
        Utils.parse(parser, args);
    
        // val context = Utils.getContext()
        String modelPathPrefix = parser.getOptions(modelPathPrefix);
        String inputImagePath = parser.inputImagePath;
        int batchSize = parser.batchSize;
        int times = parser.times;
    
        /*
        NDArrayCollector.auto().withScope {
            println("Running single inference")
            val inferenceTimeSingle = runSingleInference(modelPathPrefix, inputImagePath, context, 1, times)
        
            Utils.printStatistics(inferenceTimeSingle, "single_inference")
        }
        */
    }
}
