package mxnet;

import org.apache.mxnet.infer.javaapi.*;
import org.apache.mxnet.javaapi.*;
import org.apache.mxnet.ResourceScope;
import org.kohsuke.args4j.Option;

import java.awt.image.BufferedImage;
import org.kohsuke.args4j.CmdLineParser;

import java.util.ArrayList;
import java.util.List;

public class SSDObjectDetection {
    @Option(name = "--modelPathPrefix", usage = "The model to benchmark")
    private static String modelPathPrefix = "/tmp/resnet50_ssd/resnet50_ssd_model";
    @Option(name = "--inputImagePath", usage = "Input image path for single inference")
    private static String inputImagePath = "/tmp/resnet50_ssd/images/dog.jpg";
    @Option(name = "--batchSize", usage = "Batchsize of the model")
    private static int batchSize = 1;
    @Option(name = "--times", usage = "Number of times to run the benchmark")
    private static int times = 1;
    
    ObjectDetector loadModel(String modelPathPrefix, List<Context> context, int batchSize) {
        Shape inputShape = new Shape(new int[] {batchSize, 3, 512, 512});
        List<DataDesc> inputDescriptors = new ArrayList<>();
        inputDescriptors.add(new DataDesc("data", inputShape, DType.Float32(), "NCHW"));
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
    
    double[] runInference(String modelPathPrefix, String inputImagePath, List<Context> context, int batchSize, int times) {
        ObjectDetector loadedModel = loadModel(modelPathPrefix, context, batchSize);
        List<Double> inferenceTimes = new ArrayList<Double>();
        List<NDArray> dataset = new ArrayList<>();
        dataset.add(loadSingleImage(inputImagePath, context));
        
        // Warm up intervals
        // println("Warming up the system")
        for (int i = 0; i < 5; i++) {
            try(ResourceScope scope = new ResourceScope()) {
                List<List<ObjectDetectorOutput>> output = loadedModel.objectDetectWithNDArray(dataset, 3);
            }
        }
        // println("Warm up done")
        
        double[] result = new double[times];
        for (int i = 0; i < times; i++) {
            try(ResourceScope scope = new ResourceScope()) {
                long startTime = System.nanoTime();
                List<List<ObjectDetectorOutput>> output = loadedModel.objectDetectWithNDArray(dataset, 3);
                result[i] = (System.nanoTime() - startTime) / (1e6 * 1.0);
                System.out.printf("Inference time at iteration: %d is : %f \n", i, result[i]);
            }
        }
        return result;
    }
    
    
    public static void main(String[] args) {
        SSDObjectDetection inst = new SSDObjectDetection();
        CmdLineParser parser = Utils.parse(inst, args);
        
        List<Context> context = Utils.getContext();
    
        try(ResourceScope scope = new ResourceScope()) {
            System.out.println("Running single inference");
            double[] inferenceTimes = inst.runInference(modelPathPrefix, inputImagePath, context, 1, times);
            Utils.printStatistics(inferenceTimes, "single_inference");
        }
        try(ResourceScope scope = new ResourceScope()) {
            System.out.println("Running batch inference with batsize : " + batchSize);
            double[] inferenceTimes = inst.runInference(modelPathPrefix, inputImagePath, context, batchSize, times);
            Utils.printStatistics(inferenceTimes, "single_inference");
        }
    
        try(ResourceScope scope = new ResourceScope()) {
            System.out.println("Running batch inference with batsize : " + 2 * batchSize);
            double[] inferenceTimes = inst.runInference(modelPathPrefix, inputImagePath, context, 2 * batchSize, times);
            Utils.printStatistics(inferenceTimes, "single_inference");
        }
    
        try(ResourceScope scope = new ResourceScope()) {
            System.out.println("Running batch inference with batsize : " + 4 * batchSize);
            double[] inferenceTimes = inst.runInference(modelPathPrefix, inputImagePath, context, 4 * batchSize, times);
            Utils.printStatistics(inferenceTimes, "single_inference");
        }
    }
}
