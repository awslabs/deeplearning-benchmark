/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package mxnet;

import org.apache.mxnet.ResourceScope;
import org.apache.mxnet.javaapi.*;
import org.apache.mxnet.javaapi.Context;
import org.apache.mxnet.javaapi.NDArray;
import org.apache.mxnet.infer.javaapi.Predictor;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * Benchmark resnet18 and resnet_end_to_end model for single / batch inference
 * and CPU / GPU
 */
public class EndToEndModelWoPreprocessing {
    static NDArray$ NDArray = NDArray$.MODULE$;

    @Option(name = "--model-path-prefix", usage = "input model directory and prefix of the model")
    private String modelPathPrefix = "resnet18_v1";
    @Option(name = "--num-runs", usage = "Number of runs")
    private int numOfRuns = 1;
    @Option(name = "--batchsize", usage = "batch size")
    private int batchSize = 25;
    @Option(name = "--end-to-end", usage = "benchmark with e2e / non e2e")
    private boolean isE2E = false;
    @Option(name = "--warm-up", usage = "warm up iteration")
    private int timesOfWarmUp = 5;
    @Option(name = "--use-gpu", usage = "use gpu or cpu")
    private boolean useGPU = false;

    // process the image explicitly Resize -> ToTensor -> Normalize
    private static NDArray preprocessImage(NDArray nd) {
        NDArray resizeImg;
        NDArray[] arr = new NDArray[nd.shape().get(0)];
        for (int i = 0; i < nd.shape().get(0); i++) {
            arr[i] = Image.imResize(nd.at(i), 224, 224);
        }
        resizeImg = NDArray.stack(arr, 0, arr.length, null)[0];

        resizeImg = NDArray.cast(resizeImg, "float32", null)[0];
        resizeImg = resizeImg.div(255.0);
        NDArray totensorImg = (NDArray.swapaxes(NDArray.new swapaxesParam(resizeImg).setDim1(1).setDim2(3)))[0];
        // use 0.456 instead of (0.485, 0.456, 0.406) to simply the logic
        totensorImg = totensorImg.subtract(0.456);
        // use 0.224 instead of 0.229, 0.224, 0.225 to simply the logic
        NDArray preprocessedImg = totensorImg.div(0.224);

        return preprocessedImg;
    }

    private static void printAvg(double[] inferenceTimes, String metricsPrefix, int batchSize)  {
        double sum = 0.0;
        for (double time: inferenceTimes) {
            sum += time;
        }
        double average = sum / (batchSize * inferenceTimes.length);
        System.out.println(String.format("%s_average %1.2fms",metricsPrefix, average));
    }

    private static void runInference(String modelPathPrefix, List<Context> context, int batchSize, boolean isE2E, int numOfRuns, int timesOfWarmUp) {
        Shape inputShape;
        List<DataDesc> inputDescriptors = new ArrayList<>();
        if (isE2E) {
            inputShape = new Shape(new int[]{1, 300, 300, 3});
            inputDescriptors.add(new DataDesc("data", inputShape, DType.UInt8(), "NHWC"));
        } else {
            inputShape = new Shape(new int[]{1, 3, 224, 224});
            inputDescriptors.add(new DataDesc("data", inputShape, DType.Float32(), "NCHW"));
        }
        Predictor predictor = new Predictor(modelPathPrefix, inputDescriptors, context,0);

        double[] times = new double[numOfRuns];

        for (int n = 0; n < numOfRuns + timesOfWarmUp; n++) {
            try(ResourceScope scope = new ResourceScope()) {
                NDArray nd = NDArray.random_uniform(
                        NDArray.new random_uniformParam()
                                .setLow(0f)
                                .setHigh(255f)
                                .setShape(new Shape(new int[]{batchSize, 300, 300, 3})))[0];

                NDArray img = NDArray.cast(nd, "uint8", null)[0];
                NDArray imgWithBatchNum;
                NDArray preprocessedImage = null;
                Long curretTime = 0l;
                // time the latency after warmup
                if (n >= timesOfWarmUp) {
                    curretTime = System.nanoTime();
                }
                if (isE2E) {
                    preprocessedImage = img;
                } else {
                    preprocessedImage = preprocessImage(img);
                }
                preprocessedImage.asInContext(context.get(0));
                List<NDArray> input = new ArrayList<>();
                input.add(preprocessedImage);
                List<NDArray> output = predictor.predictWithNDArray(input);
                output.get(0).waitToRead();
                if (n >= timesOfWarmUp) {
                    times[n - timesOfWarmUp] = (System.nanoTime() - curretTime) / (1e6 * 1.0);
                }
            }

        }
        System.out.print((isE2E) ? "E2E " : "Non E2E ");
        printAvg(times, (batchSize > 1) ? "batch_inference" : "single_inference", batchSize);
    }

    public static void main(String[] args) {
        EndToEndModelWoPreprocessing inst = new EndToEndModelWoPreprocessing();
        CmdLineParser parser  = new CmdLineParser(inst);
        try {
            parser.parseArgument(args);
        } catch (Exception e) {
            System.out.println(e.getMessage());
            parser.printUsage(System.err);
            System.exit(1);
        }

        List<Context> context = new ArrayList<Context>();
        context.add((inst.useGPU) ? Context.gpu() : Context.cpu());

        runInference(inst.modelPathPrefix, context, inst.batchSize, inst.isE2E, inst.numOfRuns, inst.timesOfWarmUp);
    }
}
