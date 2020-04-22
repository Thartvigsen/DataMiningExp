import torch
from torch import nn

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = self.order_sim
        else:
            self.sim = self.cosine_sim

        self.max_violation = max_violation

    def cosine_sim(self, im, s):
        """Cosine similarity between all the image and sentence pairs
        """
        return im.mm(s.t())

    def order_sim(self, im, s):
        """Order embeddings similarity measure $max(0, s-im)$
        """
        YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
               - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
        score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
        return score

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

class LSTEP():
    def __init__(self):
        pass

    def pairwise_sub(self, first_tensor, second_tensor):
        column = first_tensor.unsqueeze(2)
        row = second_tensor.unsqueeze(1)
        return column - row

    def pairwise_and(self, first_tensor, second_tensor):
        column = first_tensor.unsqueeze(2)
        row = second_tensor.unsqueeze(1)
        return column & row

    def lsep(self, y_hat, y):
        shape = y.shape
        y_i = torch.eq(y.float(), torch.ones(shape))
        y_i_bar = torch.ne(y.float(), torch.ones(shape))

        # get indices to check
        truth_matrix = self.pairwise_and(y_i, y_i_bar).float()

        # calculate all exp'd differences
        sub_matrix = self.pairwise_sub(y_hat, y_hat)
        exp_matrix = torch.exp(-sub_matrix)

        # check which differences to consider and sum them
        sparse_matrix = exp_matrix * truth_matrix
        sums = torch.sum(sparse_matrix, dim=(1, 2))

        # get normalizing terms and apply them
        y_i_sizes = torch.sum(y_i.float(), dim=1)
        y_i_bar_sizes = torch.sum(y_i_bar.float(), dim=1)
        normalizers = y_i_sizes * y_i_bar_sizes + 1e-07 # Add epsilon to avoid div by 0
        results = sums / normalizers
        return results.mean(0)  # Average over the batch

    def forward(self, y_bar, y_hat, y):
        loss = []
        for t in range(len(y_bar)):
            # For each step of the class prediction sequence, compute the LSEP for the appropriate classes
            loss.append(self.lsep(y_bar*y_hat, y_bar*y)) # Convert un-predicted to 0s (is this right?)

class LSEP(nn.Module):
    def __init__(self):
        pass

    def pairwise_sub(self, first_tensor, second_tensor):
        column = first_tensor.unsqueeze(2)
        row = second_tensor.unsqueeze(1)
        return column - row

    def pairwise_and(self, first_tensor, second_tensor):
        column = first_tensor.unsqueeze(2)
        row = second_tensor.unsqueeze(1)
        return column & row                                                                                                               

    def forward(self, y_hat, y):
        shape = y.shape
        y_in = torch.eq(y.float(), torch.ones(shape))
        y_out = torch.ne(y.float(), torch.ones(shape))

        # get indices to check   
        truth_matrix = self.pairwise_and(y_in, y_out).float()

        # calculate all exp'd differences
        sub_matrix = self.pairwise_sub(y_hat, y_hat)
        exp_matrix = torch.exp(-sub_matrix)

        # check which differences to consider and sum them
        sparse_matrix = exp_matrix * truth_matrix
        sums = 1 + torch.sum(sparse_matrix, dim=(1, 2))
        
    #     # get normalizing terms and apply them
    #     normalizers = getNormalizers(y_in, y_out)
    #     results = torch.log(sums)/normalizers
    #     return results.mean(0) # Mean over the batch (?)

        return torch.log(sums).mean() # Average over batch
