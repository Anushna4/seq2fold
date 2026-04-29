from prediction_utils import predict_fold


def predict_sequence(seq):
    return predict_fold(seq)["prediction"]


seq = (
    "SLFEQLGGQAAVQAVTAQFYANIQADATVATFFNGIDMPNQTNKTAAF"
    "LCAALGGPNAWTGRNLKEVHANMGVSNAQFTTVIGHLRSALTGAGVAAALVEQTVAVAETVRGDVVTV"
)

print("Prediction:", predict_sequence(seq))
