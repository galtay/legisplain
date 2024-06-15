bash ./download.sh &> "logs/download-$(date +%s).log"
bash ./sync.sh &> "logs/sync-$(date +%s).log"
