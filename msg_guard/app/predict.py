from model import GRU
from data_preprocessing import MsgPredict
import torch

def prediction(msg_body):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load("/app/datas/artifacts/model.pt", map_location=device)

    mgs_predict = MsgPredict(msg_body.dict())
    features = mgs_predict.SetData()

    if features != -1:
        model.eval()
        with torch.no_grad():
            output = model(features)
            print(output)
            prediction = output.argmax(dim=1, keepdim=True)
            
        print(prediction.item())
        return prediction.item()
    else:
        return 2