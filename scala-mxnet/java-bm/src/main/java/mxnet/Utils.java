package mxnet;

import org.apache.mxnet.javaapi.Context;
import org.kohsuke.args4j.CmdLineParser;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Utils {
    
    private static boolean runBatch = false;
    
    public static void parse(Object inst, String[] args) {
        CmdLineParser parser  = new CmdLineParser(inst);
        try {
            parser.parseArgument(args);
        } catch (Exception e) {
            System.err.println(e.getMessage() + e);
            parser.printUsage(System.err);
            System.exit(1);
        }
    }
    
    private static long percentile(int p, long[] seq) {
        Arrays.sort(seq);
        int k = (int) Math.ceil((seq.length - 1) * (p / 100.0));
        return seq[k];
    }
    
    private static void printStatistics(long[] inferenceTimesRaw, String metricsPrefix)  {
        long[] inferenceTimes = inferenceTimesRaw;
        // remove head and tail
        if (inferenceTimes.length > 2) {
            inferenceTimes = Arrays.copyOfRange(inferenceTimesRaw,
                    1, inferenceTimesRaw.length - 1);
        }
        double p50 = percentile(50, inferenceTimes) / 1.0e6;
        double p99 = percentile(99, inferenceTimes) / 1.0e6;
        double p90 = percentile(90, inferenceTimes) / 1.0e6;
        long sum = 0;
        for (long time: inferenceTimes) sum += time;
        double average = sum / (inferenceTimes.length * 1.0e6);
        
        System.out.println(
                String.format("\n%s_p99 %fms\n%s_p90 %fms\n%s_p50 %fms\n%s_average %1.2fms",
                        metricsPrefix, p99, metricsPrefix, p90,
                        metricsPrefix, p50, metricsPrefix, average)
        );
    }
}