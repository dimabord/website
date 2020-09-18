__author__ = 'dimabord'
import pickle



class SentimentClassifier(object):
    def __init__(self):
        with open('sentiment_analysis/week7_model.pickle', 'rb') as f:
            self.model = pickle.load(f)
        self.classes_dict = {0: "отрицательный", 1: "положительный", -1: "ошибка"}

    @staticmethod
    def get_probability_words(probability):
        if probability < 0.55:
            return "возможно"
        if probability < 0.7:
            return "скорее всего"
        if probability > 0.95:
            return "точно"
        else:
            return ""

    def predict_text(self, text):
        text = [text]
        try:
            return self.model.predict(text)[0], self.model.decision_function(text)[0].max()
        except:
            print("ошибка")
            return -1, 0.8



    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        class_prediction = prediction[0]
        prediction_probability = prediction[1]
        return self.get_probability_words(prediction_probability) + " " + self.classes_dict[class_prediction]