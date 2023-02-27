from model import GRU
from data_preprocessing import MsgPredict
import torch

def predict(msg_body):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load("artifacts/model.pt", map_location=device)

    # sample = {
    #     "id": 3333,
    #     "msg_body": "취업지원관과 함께하는 적성검사만족도 조사"
    # }

    MsgPredict = MsgPredict(msg_body)
    features = MsgPredict.SetData()


    model.eval()
    with torch.no_grad():
        output = model(features)
        print(output)
        prediction = output.argmax(dim=1, keepdim=True)
        
    print(prediction)
    # if prediction == 1:
    #     print("This Msg is Normal")
    # else:
    #     print("This Msg is Spam")
    return prediction