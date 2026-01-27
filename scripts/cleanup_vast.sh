#!/bin/bash
# Wait for download to complete, then destroy instance

TARBALL="/root/distillix/models/glm-4.7-flash-int4.tar.gz"
EXPECTED_SIZE=14000000000  # ~14GB
API_KEY="9b698b6991d82f28d7404a53c70267cef1769bcb3c339e7391c9f718dd755725"
INSTANCE_ID="30521054"

echo "Monitoring download..."

while true; do
    if [ -f "$TARBALL" ]; then
        SIZE=$(stat -c%s "$TARBALL" 2>/dev/null || echo 0)
        if [ "$SIZE" -ge "$EXPECTED_SIZE" ]; then
            echo "Download complete! Size: $SIZE"
            
            # Verify tarball
            if tar -tzf "$TARBALL" > /dev/null 2>&1; then
                echo "Tarball verified OK"
                
                # Destroy instance
                echo "Destroying Vast.ai instance $INSTANCE_ID..."
                curl -s -X DELETE "https://console.vast.ai/api/v0/instances/${INSTANCE_ID}/?api_key=${API_KEY}"
                echo "Instance destroyed. Billing stopped."
                
                # Extract
                cd /root/distillix/models/
                echo "Extracting model..."
                tar -xzf glm-4.7-flash-int4.tar.gz
                echo "Done!"
                exit 0
            else
                echo "Tarball corrupt, waiting..."
            fi
        fi
        echo "$(date): Downloaded $(( SIZE / 1024 / 1024 ))MB / 14336MB"
    fi
    sleep 30
done
