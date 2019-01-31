#!/usr/bin/env bash

set -e

POSITIONAL=()

while [[ $# -gt 0 ]]
do
    key="$1"
    case ${key} in
        -u|--url)
        URL="$2"
        shift
        shift
        ;;
        -d|--image)
        IMAGE="$2"
        shift
        shift
        ;;
        -c|--concurrency)
        CONCURRENCY="$2"
        shift
        shift
        ;;
        -n|--requests)
        REQUESTS="$2"
        shift
        shift
        ;;
        -i|--input)
        INPUT="$2"
        shift
        shift
        ;;
        -w|--worker)
        WORKER="$2"
        shift
        shift
        ;;
        -s|--s3)
        UPLOAD="$2"
        shift
        ;;
        --default)
        DEFAULT=YES
        shift
        ;;
        *)
        POSITIONAL+=("$1")
        shift
        ;;
    esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters


if [[ -z "${URL}" ]]; then
    echo "URL is required, for example:"
    echo "benchmark.sh -u https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet50v1.mar"
    echo "benchmark.sh -i lstm.json -u https://s3.amazonaws.com/model-server/model_archive_1.0/lstm_ptb.mar"
    echo "benchmark.sh -c 500 -n 50000 -i noop.json -u https://s3.amazonaws.com/model-server/model_archive_1.0/noop-v1.0.mar"
    echo "benchmark.sh -d local-image -u https://s3.amazonaws.com/model-server/model_archive_1.0/noop-v1.0.mar"
    exit 1
fi

if [[ -x "$(command -v nvidia-docker)" ]]; then
    GPU=true
else
    GPU=false
fi

if [[ "${GPU}" == "true" ]]; then
    DOCKER_RUNTIME="--runtime=nvidia"
    if [[ -z "${IMAGE}" ]]; then
        IMAGE=awsdeeplearningteam/mxnet-model-server:nightly-mxnet-gpu
        docker pull "${IMAGE}"
    fi
    HW_TYPE=gpu
else
    if [[ -z "${IMAGE}" ]]; then
        IMAGE=awsdeeplearningteam/mxnet-model-server:nightly-mxnet-cpu
        docker pull "${IMAGE}"
    fi
    HW_TYPE=cpu
fi

if [[ -z "${CONCURRENCY}" ]]; then
    CONCURRENCY=100
fi

if [[ -z "${REQUESTS}" ]]; then
    REQUESTS=1000
fi

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILENAME="${URL##*/}"
MODEL="${FILENAME%.*}"

rm -rf /tmp/benchmark
mkdir -p /tmp/benchmark/conf
mkdir -p /tmp/benchmark/logs
cp -f ${BASEDIR}/config.properties /tmp/benchmark/conf/config.properties
echo "" >> /tmp/benchmark/conf/config.properties
echo "load_models=benchmark=${URL}" >> /tmp/benchmark/conf/config.properties
if [[ ! -z "${WORKER}" ]]; then
    echo "default_workers_per_model=${WORKER}" >> /tmp/benchmark/conf/config.properties
fi

if [[ ! -z "${INPUT}" ]] && [[ -f "${BASEDIR}/${INPUT}" ]]; then
    CONTENT_TYPE="application/json"
    cp -rf ${BASEDIR}/${INPUT} /tmp/benchmark/input
else
    CONTENT_TYPE="application/jpg"
    curl https://s3.amazonaws.com/model-server/inputs/kitten.jpg -s -S -o /tmp/benchmark/input
fi

# start mms docker
set +e
docker rm -f mms
set -e
docker run ${DOCKER_RUNTIME} --name mms -p 8080:8080 -p 8081:8081 \
    -v /tmp/benchmark/conf:/opt/ml/conf \
    -v /tmp/benchmark/logs:/home/model-server/logs \
    -u root -itd ${IMAGE} mxnet-model-server --start \
    --mms-config /opt/ml/conf/config.properties

MMS_VERSION=`docker exec -it mms pip freeze | grep mxnet-model-server`

until curl -s "http://localhost:8080/ping" > /dev/null
do
  echo "Waiting for docker start..."
  sleep 3
done

sleep 10

result_file="/tmp/benchmark/result.txt"
metric_log="/tmp/benchmark/logs/model_metrics.log"

ab -c ${CONCURRENCY} -n ${REQUESTS} -k -p /tmp/benchmark/input -T "${CONTENT_TYPE}" \
    http://127.0.0.1:8080/predictions/benchmark > ${result_file}

line50=$((${REQUESTS} / 2))
line90=$((${REQUESTS} * 9 / 10))
line99=$((${REQUESTS} * 99 / 100))

grep "PredictionTime" ${metric_log} | cut -c55- | cut -d"|" -f1 | sort -g > /tmp/benchmark/predict.txt
grep "PreprocessTime" ${metric_log} | cut -c55- | cut -d"|" -f1 | sort -g > /tmp/benchmark/preprocess.txt
grep "InferenceTime" ${metric_log} | cut -c54- | cut -d"|" -f1 | sort -g > /tmp/benchmark/inference.txt
grep "PostprocessTime" ${metric_log} | cut -c56- | cut -d"|" -f1 | sort -g > /tmp/benchmark/postprocess.txt

MODEL_P50=`sed -n "${line50}p" /tmp/benchmark/predict.txt`
MODEL_P90=`sed -n "${line90}p" /tmp/benchmark/predict.txt`
MODEL_P99=`sed -n "${line99}p" /tmp/benchmark/predict.txt`

MMS_ERROR=`grep "Failed requests:" ${result_file} | awk '{ print $NF }'`
MMS_TPS=`grep "Requests per second:" ${result_file} | awk '{ print $4 }'`
MMS_P50=`grep " 50\% " ${result_file} | awk '{ print $NF }'`
MMS_P90=`grep " 90\% " ${result_file} | awk '{ print $NF }'`
MMS_P99=`grep " 99\% " ${result_file} | awk '{ print $NF }'`
MMS_MEAN=`grep -E "Time per request:.*mean\)" ${result_file} | awk '{ print $4 }'`
MMS_ERROR_RATE=`echo "scale=2;100 * ${MMS_ERROR}/${REQUESTS}" | bc | awk '{printf "%f", $0}'`

echo "" > /tmp/benchmark/report.txt
echo "======================================" >> /tmp/benchmark/report.txt
curl -s http://localhost:8081/models/benchmark >> /tmp/benchmark/report.txt
echo "Inference result:" >> /tmp/benchmark/report.txt
curl -s -X POST http://127.0.0.1:8080/predictions/benchmark -H "Content-Type: ${CONTENT_TYPE}" \
    -T /tmp/benchmark/input >> /tmp/benchmark/report.txt
echo "" >> /tmp/benchmark/report.txt
echo "" >> /tmp/benchmark/report.txt

echo "======================================" >> /tmp/benchmark/report.txt
echo "MMS version: ${MMS_VERSION}" >> /tmp/benchmark/report.txt
echo "CPU/GPU: ${HW_TYPE}" >> /tmp/benchmark/report.txt
echo "Model: ${MODEL}" >> /tmp/benchmark/report.txt
echo "Concurrency: ${CONCURRENCY}" >> /tmp/benchmark/report.txt
echo "Requests: ${REQUESTS}" >> /tmp/benchmark/report.txt

echo "Model latency P50: ${MODEL_P50}" >> /tmp/benchmark/report.txt
echo "Model latency P90: ${MODEL_P90}" >> /tmp/benchmark/report.txt
echo "Model latency P99: ${MODEL_P99}" >> /tmp/benchmark/report.txt
echo "MMS throughput: ${MMS_TPS}" >> /tmp/benchmark/report.txt
echo "MMS latency P50: ${MMS_P50}" >> /tmp/benchmark/report.txt
echo "MMS latency P90: ${MMS_P90}" >> /tmp/benchmark/report.txt
echo "MMS latency P99: ${MMS_P99}" >> /tmp/benchmark/report.txt
echo "MMS latency mean: ${MMS_MEAN}" >> /tmp/benchmark/report.txt
echo "MMS error rate: ${MMS_ERROR_RATE}%" >> /tmp/benchmark/report.txt

cat /tmp/benchmark/report.txt

if [[ ! -z "${UPLOAD}" ]]; then
    TODAY=`date +"%y-%m-%d_%H"`
    echo "Saving on S3 bucket on s3://benchmarkai-metrics-prod/daily/mms/${HW_TYPE}/${TODAY}/${MODEL}"

    aws s3 cp /tmp/benchmark/ s3://benchmarkai-metrics-prod/daily/mms/${HW_TYPE}/${TODAY}/${MODEL} --recursive

    echo "Files uploaded"
fi
