import torch
import torch.nn as nn

bce, sigmoid, softmax = nn.BCELoss(), nn.Sigmoid(), nn.Softmax(dim=1)

class ASM2VEC(nn.Module):
    def __init__(self, vocab_size, function_size, embedding_size):
        super(ASM2VEC, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.embeddings_f = nn.Embedding(function_size, 2 * embedding_size)
        self.embeddings_r = nn.Embedding(vocab_size, 2 * embedding_size)
        
    def v(self, inp):
        e  = self.embeddings(inp[:,1:])
        v_f = self.embeddings_f(inp[:,0])
        v_prev = torch.cat([e[:,0], (e[:,1] + e[:,2]) / 2], dim=1)
        v_next = torch.cat([e[:,3], (e[:,4] + e[:,5]) / 2], dim=1)
        v = ((v_f + v_prev + v_next) / 3).unsqueeze(2)
        return v

    def forward(self, inp, pos, neg):
        device, batch_size = inp.device, inp.shape[0]
        v = self.v(inp)
        # negative sampling loss
        pred = torch.bmm(self.embeddings_r(torch.cat([pos, neg], dim=1)), v).squeeze()
        label = torch.cat([torch.ones(batch_size, 3), torch.zeros(batch_size, neg.shape[1])], dim=1).to(device)
        return bce(sigmoid(pred), label)

    def predict(self, inp, pos):
        device, batch_size = inp.device, inp.shape[0]
        v = self.v(inp)
        probs = torch.bmm(self.embeddings_r(torch.arange(self.embeddings_r.num_embeddings).repeat(batch_size, 1).to(device)), v).squeeze(dim=2)
        return softmax(probs)
