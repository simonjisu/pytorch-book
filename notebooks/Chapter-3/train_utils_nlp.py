import torch
import torch.nn.functional as F 

def train(model, train_loader, loss_func, optimizer, step, print_step=200):
    """train function"""
    model.train()
    for i, batch in enumerate(train_loader):
        inputs, targets = batch.text, batch.label.float()
        # 경사 초기화
        optimizer.zero_grad()
        # 순방향 전파
        outputs = model(inputs)
        # 손실값 계산
        loss = loss_func(outputs, targets)
        # 역방향 전파
        loss.backward()
        # 매개변수 업데이트
        optimizer.step()
    
        if i % print_step == 0:
            print('Train Step: {} ({:05.2f}%)  \tLoss: {:.4f}'.format(
                    step, 100.*(i*train_loader.batch_size)/len(train_loader.dataset), 
                    loss.item()))

def test(model, test_loader, loss_func):
    """test function"""
    # 모델에게 평가단계이라고 선언함
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch.text, batch.label.float()
            # 순방향전파
            outputs = model(inputs)
            # 손실값 계산(합)
            test_loss += loss_func(outputs, targets, reduction="sum").item()
            # 예측값
            preds = torch.sigmoid(outputs).ge(0.5).float()
            # 정확하게 예측한 개수를 기록한다
            correct += preds.eq(targets).sum().item()
            
    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:05.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset), 100. * test_acc))
    return test_loss, test_acc

def main(model, train_loader, test_loader, loss_func, optimizer, n_step, 
         save_path=None, print_step=30):
    """메인 학습 함수"""
    test_accs = []
    best_acc = 0.0

    for step in range(1, n_step+1):
        # 훈련 단계
        train(model, train_loader, loss_func, optimizer, 
              step=step, print_step=print_step)
        # 평가 단계
        test_loss, test_acc = test(model, test_loader, 
                                   loss_func=F.binary_cross_entropy_with_logits)
        # 테스트 정확도 기록
        test_accs.append(test_acc)
        # 모델 최적의 매개변수값을 저장할지 결정하고 기록한다.
        if len(test_accs) >= 2:
            if test_acc >= best_acc:
                best_acc = test_acc
                best_state_dict = model.state_dict()
                print("discard previous state, best model state saved!")
        print("")

    # 매개변수 값 저장하기
    if save_path is not None:
        torch.save(best_state_dict, save_path)