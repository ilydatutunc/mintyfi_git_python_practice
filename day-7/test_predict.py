import pytest
from predict_refactor import predict_message

def test_predict_spam():
    msg =  "Congratulations!!! You've won a $1000 Walmart gift card. Click to claim now."
    assert predict_message(msg) == "spam"

def test_predict_ham():
    msg = "Hey Ilayda, meeting at 3pm is still on right?"
    assert predict_message(msg) == "ham"

def test_predict_empty():
    with pytest.raises(ValueError):
        predict_message("")
