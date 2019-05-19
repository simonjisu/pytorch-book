import torch
import torch.nn as nn

class Encoder(nn.Module):
    """Encoder"""
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers, bidirec=False):
        """
        args:
         - vocab_size: 소스 단어장 크기
         - embed_size: 인코더 임베딩 크기
         - hidden_size: 인코더 히든 크기 
         - n_layers: 인코더 층의 깊이
        """
        super(Encoder, self).__init__()        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_direc = 2 if bidirec else 1
        # 인코더 임베딩 층
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 인코더 RNN 층
        self.gru = nn.GRU(embed_size, 
                          hidden_size, 
                          n_layers, 
                          bidirectional=bidirec, 
                          batch_first=True)

    def forward(self, inputs):
        """
        입력의 크기는 다음과 같으며 각 심볼이 의미하는 바는 다음과 같다.
        Inputs:
        - inputs: B, T_e
        Outputs:
        - outputs: B, T_e, n_directions*H_e
        - hiddens: 1, B, n_directions*H_e
        ==========================================
        B: 미니배치 크기
        T_e: 인코더에 입력된 문장의 최대 길이
        E_e: 인코더 임베딩 크기
        H_e: 인코더 은닉층 크기
        """
        # 임베딩 된 텐서의 크기 변화: (B, T_e) > (B, T_e, E_e)
        embeded = self.embedding(inputs)
        
        # gru 출력의 크기, output 은 안쓰이기 때문에 _ 에다 저장해주었다.
        # output: (B, T_e, n_directions*H_e)
        # hidden: (n_layers*n_directions, B, H_e)
        _, hidden = self.gru(embeded)
        
        # 마지막 층의 은닉 상태를 가져오고 인코더의 출력으로 전달한다.
        # 크기: (1, B, n_directions*H)
        last_hidden = torch.cat([h for h in hidden[-self.n_direc:]], 1)
        
        return last_hidden.unsqueeze(0)
    
    
class Decoder(nn.Module):
    """Decoder"""
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers, sos_idx=2):
        """
        args:
         - vocab_size: 타겟 단어장 크기
         - embed_size: 디코더 임베딩 크기
         - hidden_size: 디코더 히든 크기 = "인코더 히든 크기 * 인코더의 RNN 방향 개수" 로 설정한다.
         - n_layers: 디코더 층의 깊이
         - sos_idx: 타겟 단어장에서 시작 토큰의 인덱스
        """
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.sos_idx = sos_idx
        # 디코더 임베딩 층
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 디코더 RNN 층
        self.gru = nn.GRU(embed_size+hidden_size, hidden_size, n_layers, bidirectional=False, 
                          batch_first=True)
        # 선형결합층
        self.linear = nn.Linear(embed_size+2*hidden_size, vocab_size, bias=False)

    def init_sos(self, batch_size, device):
        # 시작을 알리는 <s> 토큰을 텐서로 생성한다. 크기는 (B, 1)
        return torch.LongTensor([self.sos_idx]*batch_size).unsqueeze(1).to(device)
    
    def forward(self, hiddens, maxlen=None, eos_idx=None):
        """
        입력의 크기는 다음과 같으며 각 심볼이 의미하는 바는 다음과 같다.
        Inputs:
        - hiddens: 1, B, n_direction(encoder)*H_d 
        - max_len: T_d
        - eos_idx: 테스트 용도
        Outputs:
        - scores: results of all predictions = B, T_d, vocab_size
        ==========================================
        B: 미니배치 크기
        T_d: 디코더에 입력된 문장의 최대 길이
        E_d: 디코더 임베딩 크기
        H_d: 디코더 은닉층 크기
        """
        maxlen = 32 if maxlen is None else maxlen
        
        # 시작 토큰으로 디코더 입력값을 초기화 한다. 크기는 (B, 1)
        inputs = self.init_sos(hiddens.size(1), device=hiddens.device)
        
        # 임베딩 층을 통과한다. 크기는 (B, 1, E_d)
        embeded = self.embedding(inputs)
        
        # 인코더에서 가져온 은닉층을 디코더의 초기화 값으로 적용하기 위해 디코더 RNN 층의 개수로 맞춰준다.
        # 크기 변화: (1, B, H_d) > (n_layers, B, H_d)
        if hiddens.size(0) != self.n_layers: 
            hiddens = hiddens.repeat(self.n_layers, 1, 1)
        
        # 손실함수에 전달하기 위해 예측 스코어를 저장한다.
        scores = []  
        
        for i in range(1, maxlen):
            # RNN에 들어갈 입력값을 만들어준다 concat(y_{t-1}, c)
            # cat[(B, 1, E_d), (B, 1, H_d)] > (B, seq_len=1, E_d+H_d)
            inputs = torch.cat((embeded, hiddens[-1, :, :].unsqueeze(1)), dim=2)
            
            # RNN 출력 값을 얻는다. h_t = f(h{i-1}, y{i-1}, c): 
            # 크기변화 (B, 1, E_d+H_d) > (n_layers, B, H_d)
            _, hiddens = self.gru(inputs, hiddens)
            
            # 확률을 예측하기 전, 선형결합의 출력값을 얻는다
            # score = g(h{i}, y{i-1}, c)
            # 합쳐진 입력값(inputs)과 RNN의 마지막 층의 정보(hiddens)를 결합한다.
            # 크기 변화: cat[(B, E_d+H_d), (B, H_d)] > (B, E_d+H_d+H_d)
            linear_inputs = torch.cat((inputs.squeeze(1), hiddens[-1, :, :]), dim=1)
            # linear 크기 변화: (B, E_d+H_d+H_d) > (B, vocab_size)
            score = self.linear(linear_inputs)
            scores.append(score)
            
            # score를 바탕으로 다음 타겟 토큰을 예측한다.
            inputs, stop_decode = self.decode(score=score, eos_idx=eos_idx)
            if stop_decode:
                break
        # 손실함수에 전달하기 위해 텐서의 형태를 변화한다. 
        # (T_d, B, vocab_size) > (B, T_d, vocab_size)
        scores = torch.stack(scores).permute(1, 0, 2).contiguous()  
        return scores
    
    def decode(self, score, eos_idx=None):
        """
        score를 기반으로 다음 타겟 토큰을 예측한다. 
        다음 타겟 토큰을 임베딩 층을 통과시킨 값을 출력으로 반환한다
        """
        # 테스트 단계에서 디코드를 멈춰야할지 결정하는 변수, 훈련시 사용안한다.
        stop_decode = False
        
        # 다음 토큰 예측
        pred = score.softmax(-1).argmax(-1)
        
        # 다음 타겟 토큰을 임베딩 층으로 건낸다. 
        # 크기변화: (B, 1) > (B, 1, E_d)
        inputs = self.embedding(pred)
        
        if (eos_idx is not None) and (pred.view(-1).item() == eos_idx):
            stop_decode = True
        
        return inputs, stop_decode
    
    
class EncoderDecoder(nn.Module):
    """Encoder - Decoder"""
    def __init__(self, enc_vocab_size, dec_vocab_size, embed_size, hidden_size, 
                 num_layers, batch_first=True, bidirec=False, sos_idx=2, ):
        """
        단순 Encoder와 Decoder 를 연결시킨 클래스다.
        
        args:
         - enc_vocab_size: 소스 단어장 크기
         - dec_vocab_size: 타겟 단어장 크기
         - embed_size: 임베딩 크기
         - hidden_size: RNN 은닉층 크기
         - num_layers: RNN 층수
         - batch_first: 미니배치 크기가 텐서의 제일 앞에 오는 지의 여부
         - bidirec: 인코더 RNN 층의 양방향 여부
         - sos_idx: 
        """
        super(EncoderDecoder, self).__init__()
        n_direct = 2 if bidirec else 1
        self.encoder = Encoder(vocab_size=enc_vocab_size, 
                               embed_size=embed_size, 
                               hidden_size=hidden_size, 
                               n_layers=num_layers, 
                               bidirec=bidirec)
        self.decoder = Decoder(vocab_size=dec_vocab_size, 
                               embed_size=embed_size, 
                               hidden_size=n_direct*hidden_size, 
                               n_layers=num_layers, 
                               sos_idx=sos_idx)
        
    def forward(self, inputs, maxlen=None, eos_idx=None):
        """
        scores 크기: (B, T_d, vocab_size)
        """
        enc_outputs = self.encoder(inputs)
        scores = self.decoder(enc_outputs, maxlen, eos_idx)
        return scores