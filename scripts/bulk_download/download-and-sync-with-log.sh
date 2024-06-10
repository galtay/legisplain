bash ./download.sh &> "logs/download-$(date +%s).log"
./sync.sh
