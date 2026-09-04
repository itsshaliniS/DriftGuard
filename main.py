import uvicorn

if __name__ == "__main__":
    print("Starting Drift Guard server on port 8000...")
    uvicorn.run("backend.api.main:app", host="127.0.0.1", port=8000, reload=False)

