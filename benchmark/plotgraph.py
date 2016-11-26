import pygal
import numpy as np
import sys
import getopt
import csv


def main(argv):

    label_list = None
    csv_list = None

    try:
        opts, args = getopt.getopt(argv, "", ["labels=","csv=","file=","runs="])
    except getopt.GetoptError:
        print("Incorrect args")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "--labels":
            label_list = arg
        elif opt == "--csv":
            csv_list = arg
        elif opt == "--file":
            out_file = arg
        elif opt == "--runs":
            num_runs = int(arg)
            
    if(label_list == None or csv_list == None or out_file == None or num_runs == None):
        print("Incorrect args")
        sys.exit(2)

    labels = label_list.split(",")
    map(str.strip, labels)

    csv_files = csv_list.split(",")
    map(str.strip, csv_files)

    line_chart = pygal.Line(logarithmic=True, truncate_legend=100, legend_at_bottom=True)
    line_chart.title = "Deep Learning Fameworks - Performance Comparison"
    
    x = np.arange(0,num_runs)
    x = np.power(2,x)
    line_chart.x_labels = map(str, x.tolist())
    
    # Add ideal plot
    ideal = np.copy(x)
    line_chart.add('Ideal', ideal.tolist() )

    index = 0
    for csv_file in csv_files:
        with open(csv_file, mode='r') as infile:
            reader = csv.reader(infile, delimiter=',')
            baseline = 0
            yval = np.empty([0])
            for row in reader:
                if(len(row) == 2):
                    if baseline == 0:
                        baseline = float(row[1])
                    yval = np.append(yval, float(row[1])/baseline)
            line_chart.add(labels[index], yval.tolist(), formatter= lambda speedup, images_per_gpu=baseline: 'Speedup: %0.2f, Images/Sec: %0.2f' % (speedup, images_per_gpu*speedup))
            index += 1


    line_chart.render_to_file(out_file)


if __name__ == "__main__":
    main(sys.argv[1:])
