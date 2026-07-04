from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_predict():
    sample = {
        "Pclass": 3,
        "Age": 30,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 10.5,
        "Sex_male": 1,
        "Embarked_Q": 0,
        "Embarked_S": 1
    }

    response = client.post(
        "/predict",
        json=sample,
        headers={"x-api-key": "my-secret-key"}
    )

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "risk_probability" in response.json()
