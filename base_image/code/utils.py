import pandas as pd

def data_parser(data):
    output = []
    for line in data.splitlines():
        try:
            text = line.decode("utf-8")
        except Exception as e:
            text = line.decode("utf-8", "ignore")
        if '"' in text:
            try:
                output.append(list(eval(text)))
            except Exception as e:
                pass
        else:
            try:
                output.append(text.split(','))
            except Exception as e:
                pass
    output = [x for x in output if len(x) == 2 and str(x[1]).isdigit()]
    if len(output) > 0:
        parsed_train = pd.DataFrame(output, columns=["text", "label"])
        parsed_train = parsed_train.astype({"text": str, "label": int})
    return parsed_train
