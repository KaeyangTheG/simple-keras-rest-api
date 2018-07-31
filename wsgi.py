from run_keras_server import app, load_api_model, load_variables

if __name__ == "__main__":
    load_api_model()
    load_variables()
    app.run()
