from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(request: Request):
    # Read JSON body
    body = await request.json()

    # Print it in the console
    print("\n==========================")
    print("✅ Received data from frontend:")
    print(json.dumps(body, indent=4))
    print("==========================\n")

    # Send a response back
    return {
        "message": "Data received successfully!",
        "received": body
    }
