import pygal
import numpy as np
import sys
import getopt
import csv
import os


def get_ssh_config_entry(line, user, pem_file_path):
    entry = "Host " + line + "\n"\
            + "\t" + "Hostname " + line + "\n" \
            + "\t" + "port 22 \n" \
            + "\t" + "user " + user + "\n" \
            + "\t" + "IdentityFile " + pem_file_path + "\n" 
    return entry


def main(argv):

    label_list = None
    csv_list = None

    try:
        opts, args = getopt.getopt(argv, "", ["hosts=","user=","pem=","out="])
    except getopt.GetoptError:
        print("Incorrect args")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "--hosts":
            hosts_file_path = os.path.abspath(os.path.expanduser(arg))
        elif opt == "--user":
            user = arg
        elif opt == "--pem":
            pem_file_path = os.path.abspath(os.path.expanduser(arg))
        elif opt == "--out":
            out_file_path = os.path.abspath(os.path.expanduser(arg))
            
    if(hosts_file_path == None or user == None or pem_file_path == None or out_file_path == None):
        print("Incorrect args")
        sys.exit(2)
        
    with open(hosts_file_path) as in_file, open(out_file_path, "w") as out_file:
        for line in in_file:
            
            line = line.strip()
            if not line:
                continue
            
            entry = get_ssh_config_entry(line, user, pem_file_path)
            
            out_file.write(entry)
            out_file.write("\n")
            



if __name__ == "__main__":
    main(sys.argv[1:])
