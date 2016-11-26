import matplotlib.pyplot as plt
import numpy as np
import getopt
import csv
import sys

def main(argv):

    label_list = None
    csv_list = None

    try:
        opts, args = getopt.getopt(argv, "", ["labels=","csv=","file="])
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


    if(label_list == None or csv_list == None or out_file == None):
        print("Incorrect args")
        sys.exit(2)

    labels = label_list.split(",")
    map(str.strip, labels)

    csv_files = csv_list.split(",")
    map(str.strip, csv_files)

    index = 0
    for csv_file in csv_files:
        with open(csv_file, mode='r') as infile:
            xval = np.array([0])
            yval = np.array([0.0])
            reader = csv.reader(infile, delimiter=',')
            for row in reader:
                if(len(row) == 2):
                    xval = np.append(xval, row[0])
                    yval = np.append(yval, row[1])
            plt.plot(xval, yval, linestyle='-', marker='+', label=labels[index])          
            index += 1

    plt.xlabel("Number of GPUs")
    plt.ylabel("Images/sec")
    plt.legend(loc="upper left")

    plt.savefig(out_file)


if __name__ == "__main__":
    main(sys.argv[1:])
