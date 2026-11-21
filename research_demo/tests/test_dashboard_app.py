import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class DashboardPageTests(unittest.TestCase):
    def test_upload_case_page_shows_inputs_without_predicting_before_submission(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        navigation = next(widget for widget in app.radio if widget.label == "Navigate")
        app = navigation.set_value("Upload case").run()

        self.assertEqual(len(app.file_uploader), 1)
        self.assertEqual(
            {widget.label for widget in app.selectbox},
            {"Grew", "Itches", "Changed", "Elevated", "Bleeds", "Hurts", "Body region"},
        )
        self.assertTrue(any(widget.label == "Age (years)" for widget in app.number_input))
        self.assertTrue(any(widget.label == "Predict and explain" for widget in app.button))
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.exception), 0)

    def test_upload_case_requires_an_image_before_prediction(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        navigation = next(widget for widget in app.radio if widget.label == "Navigate")
        app = navigation.set_value("Upload case").run()
        submit = next(widget for widget in app.button if widget.label == "Predict and explain")
        app = submit.click().run()

        self.assertTrue(
            any("Choose a PNG or JPEG image" in message.value for message in app.error)
        )
        self.assertEqual(len(app.metric), 0)

    def test_frozen_test_mel_support_is_independent_of_selected_partition(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        navigation = next(widget for widget in app.radio if widget.label == "Navigate")
        navigation.set_value("Real-data results").run()

        support = next(
            metric.value
            for metric in app.metric
            if metric.label == "MEL frozen-test support"
        )
        self.assertEqual(support, "7")

        partition = next(
            widget
            for widget in app.radio
            if widget.label == "Saved prediction partition"
        )
        partition.set_value("Frozen test").run()
        support = next(
            metric.value
            for metric in app.metric
            if metric.label == "MEL frozen-test support"
        )
        self.assertEqual(support, "7")


if __name__ == "__main__":
    unittest.main()
