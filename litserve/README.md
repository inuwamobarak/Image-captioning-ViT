# What is Litserve?
We used [Litserve](https://lightning.ai/) to handle loading an image captioning server.

Litserve provides a flexible, high-throughput serving engine for self-hosting GenAI models like LLMs and VLMs. IT allows us to batch, stream, and easily autoscale our GPU.

## How to start Litserve server:

**Start the server:** Run the script and start the server on port 8000.
**Send requests:** You can now send POST requests to the server with image_path (URL or file path) in the payload, and it will return the image caption.

The swagger UI will be available at http://0.0.0.0:8000/docs similar to FastAPI. Infact, it is built on FastAPI.
