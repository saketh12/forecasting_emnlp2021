from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from utils import GJO_Dataset, Question, Forecast, get_dataloaders
from model import LSTM_Model, LSTM_Model_With_Question

def train(device, train_loader, model, optimizer, criterion, scaler):
    sigmoid = nn.Sigmoid()
    model = model.train()
    epoch_acc = 0
    total_length = 0
    optimizer.zero_grad()
    for index, batch in tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True):
        input = batch['Input_ids'].to(device)
        attention = batch['Attention_masks'].to(device)
        predictions = batch['Forecast_predictions'].to(device)
        correct_answer = batch['Correct_answers'].to(device)
        with torch.cuda.amp.autocast():
            output = model(input, attention, predictions).to(device)
            loss = criterion((output.squeeze(0)).squeeze(1), correct_answer).to(device)
        scaler.scale(loss).backward()
        if(index + 1) % 2 == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        output = sigmoid(output)
        total_length += len(correct_answer)
        rounded_output = torch.round(output.squeeze(0).squeeze(1))
        epoch_acc += (torch.sum(rounded_output == correct_answer)).item()
    return round((epoch_acc / total_length) * 100, 3)

def evaluate(device, loader, model):
    sigmoid = nn.Sigmoid()
    model = model.eval()
    y_pred_list = []
    y_correct_list = []
    total_length = 0
    epoch_acc = 0
    with torch.no_grad():
        for index, batch in tqdm(enumerate(loader), total=len(loader), position=0, leave=True):
            input = batch['Input_ids'].to(device)
            attention = batch['Attention_masks'].to(device)
            predictions = batch['Forecast_predictions'].to(device)
            correct_answer = batch['Correct_answers'].to(device)
            output = model(input, attention, predictions).to(device)
            output = sigmoid(output)
            total_length += len(correct_answer)
            rounded_output = torch.round(output.squeeze(0).squeeze(1))
            epoch_acc += (torch.sum(rounded_output == correct_answer)).item()
            output = output.squeeze(0).squeeze(1)
            output = output.tolist()
            correct_answer = correct_answer.tolist()
            y_correct_list.extend(correct_answer)
            y_pred_list.extend(output)
    return round((epoch_acc / total_length) * 100, 3)

def experiment(daily, device):
    batch_size = 8
    learning_rate = 1e-3
    num_classes = 1
    middle_layer_shape = 256
    hidden_size = 256
    epoch_num = 0
    best_val_acc = 0
    patience = 3
    check_stopping = 0
    load = False
    if daily:
        path1 = "Daily/Weights/best_daily_text.pt"
        path2 = "Daily/Weights/current_daily_text.pt"
    else:
        path1 = "Total/Weights/best_total_text.pt"
        path2 = "Total/Weights/current_total_text.pt"
    train_loader, test_loader, val_loader = get_dataloaders(batch_size, daily, "questions.save")
    model = LSTM_Model(num_classes, middle_layer_shape, hidden_size, device).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    if load:
        load_dict = torch.load(path2)
        model.load_state_dict(load_dict['model_state_dict'])
        optimizer.load_state_dict(load_dict['optimizer_state_dict'])
    if check_stopping < patience:
        while True:
            epoch_num += 1
            train_acc = train(device, train_loader, model, optimizer, criterion, scaler)
            print("Achieved training accuracy of", train_acc)
            val_acc = evaluate(device, val_loader, model)
            print("Achieved validation accuracy of", train_acc)
            if val_acc > best_val_acc:
                check_stopping = 0
                best_val_acc = val_acc
                save_dict = {'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict()}
                torch.save(save_dict, path1)
                torch.save(save_dict, path2)
            else:
                check_stopping += 1
                save_dict = {'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict()}
                torch.save(save_dict, path2)
                print("The accuracy on the validation set does not increase")
                if check_stopping == patience:
                    print("Accuracy on validation set does not increase and has reached patience level, stop training!")
                    break
    print("Trained for", epoch_num, "epochs")
    test_acc = evaluate(device, test_loader, model)
    print("Achieved test accuracy of", test_acc)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment(True, device)