##Instructions to run scalability test on your own cluster:

1. Install MXNet with GPU support on all machines in the cluster. Installation instructions are available here: http://mxnet.io/get_started/index.html#setup-and-installation
2. Install TensorFlow with GPU support on all machines in the cluster. Installation instructions are available here: https://www.tensorflow.org/get_started/os_setup.html
3. Make sure each machine in the cluster can receive TCP connections from every other machine in the cluster in the following ports:
    - 2222
    - 2230 to (2230 + \<number_of_GPUs_in_each_machine - 1\>)
4. On a machine that has ssh access to all machines in the cluster using key-based authentication, run the following command:

    `bash runscalabilitytest.sh -m "<models>" -h <hosts-file> -g <gpus-per-machine> -n <number-of-machines> -u <user-id> -p <pem-file>`

    where,
    
    -m is a comma seperated list of models along with batch size. Model name and batch size are seperated with a ':'. Model can be 'Alexnet' or 'Inceptionv3'. Example: "Alexnet:512,Inceptionv3:32"
    
    -h is hosts file containing a newline-seperated list of machines in the cluster.
    
    -g is the number of GPUs available in each machine. 
    
    -n is the number of machines to run the tests on. This can be lower than the number of machines in hosts file.
    
    -u is the user name that can be used for ssh authentication to any of the machines in the cluster.
    
    -p is the PEM file containing the key that can be used for SSH authentication with any of the machines in the cluster.


    Example:
    
    `bash runscalabilitytest.sh -m "Alexnet:512,Inceptionv3:32" -h hosts -g 8 -n 4 -u ec2-user -p ~/my_key.pem`

5. After the script finishes the scalability tests, you can see the results in the CSV files under 'csv_*' directories. Each line in the CSV files is a key value pair where key is the number of GPUs the test was run on and value is the throughput measured as number of images processed per second. The data in the CSV files are also plotted in the graph 'comparison_graph.svg'
