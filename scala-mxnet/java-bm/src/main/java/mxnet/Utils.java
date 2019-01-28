package mxnet;

import org.apache.mxnet.javaapi.Context;
import org.kohsuke.args4j.CmdLineParser;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Utils {
    
    private static boolean runBatch = false;
    
    public static List<Context> getContext() {
        
        List<Context> context = new ArrayList<Context>();
        if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
                Integer.parseInt(System.getenv("SCALA_TEST_ON_GPU")) == 1) {
            context.add(Context.gpu());
        } else {
            context.add(Context.cpu());
        }
        return context;
    }
    
    public static CmdLineParser parse(Object inst, String[] args) {
        CmdLineParser parser  = new CmdLineParser(inst);
        try {
            parser.parseArgument(args);
        } catch (Exception e) {
            System.err.println(e.getMessage() + e);
            parser.printUsage(System.err);
            System.exit(1);
        }
        return parser;
    }
    
    private static double percentile(int p, double[] seq) {
        Arrays.sort(seq);
        int k = (int) Math.ceil((seq.length - 1) * (p / 100.0));
        return seq[k];
    }
    
    public static void printStatistics(double[] inferenceTimes, String metricsPrefix)  {
        
        double[] times = inferenceTimes;
        double p50 = percentile(50, times);
        double p99 = percentile(99, times);
        double p90 = percentile(90, times);
    
        double sum = 0;
    
        for (double i : times)
            sum += i;
        double average = sum / (times.length * 1.0);
        
        System.out.printf("\n%s_p99 %1.2f, %s_p90 %1.2f, %s_p50 %1.2f, %s_average %1.2f\n", metricsPrefix,
                p99, metricsPrefix, p90, metricsPrefix, p50, metricsPrefix, average);
        
    }
}