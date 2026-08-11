for i in $(seq 1 32); do
  curl -s http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "messages": [
        {"role": "user", "content": "请介绍一下 llama.cpp"}
      ],
      "max_tokens": 128,
      "stream": false
    }' > /tmp/llama-result-$i.json &

  if [ $((i % 8)) -eq 0 ]; then
    wait
  fi
done
wait