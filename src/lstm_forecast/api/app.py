# src/lstm_forecast/api/app.py
from lstm_forecast.api import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
