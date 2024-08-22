if __name__ == "__main__":
  import bentoml

  with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    for response in client.search(prompt='Who won the 2024 Olympics for track and field?', max_tokens=1024, num_search=10, timeout=3):
      print(response, end="", flush=True)
