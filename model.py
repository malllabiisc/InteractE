from helper import *

class InteractE(torch.nn.Module):
	"""
	Proposed method in the paper. Refer Section 6 of the paper for mode details 

	Parameters
	----------
	params:        	Hyperparameters of the model
	chequer_perm:   Reshaping to be used by the model
	
	Returns
	-------
	The InteractE model instance
		
	"""
	def __init__(self, params, chequer_perm):
		super(InteractE, self).__init__()

		self.p                  = params
		self.ent_embed		= torch.nn.Embedding(self.p.num_ent,   self.p.embed_dim, padding_idx=None); xavier_normal_(self.ent_embed.weight)
		self.rel_embed		= torch.nn.Embedding(self.p.num_rel*2, self.p.embed_dim, padding_idx=None); xavier_normal_(self.rel_embed.weight)
		self.bceloss		= torch.nn.BCELoss()

		self.inp_drop		= torch.nn.Dropout(self.p.inp_drop)
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.feature_map_drop	= torch.nn.Dropout2d(self.p.feat_drop)
		self.bn0		= torch.nn.BatchNorm2d(self.p.perm)

		flat_sz_h 		= self.p.k_h
		flat_sz_w 		= 2*self.p.k_w
		self.padding 		= 0

		self.bn1 		= torch.nn.BatchNorm2d(self.p.num_filt*self.p.perm)
		self.flat_sz 		= flat_sz_h * flat_sz_w * self.p.num_filt*self.p.perm

		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)
		self.fc 		= torch.nn.Linear(self.flat_sz, self.p.embed_dim)
		self.chequer_perm	= chequer_perm

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
		self.register_parameter('conv_filt', Parameter(torch.zeros(self.p.num_filt, 1, self.p.ker_sz,  self.p.ker_sz))); xavier_normal_(self.conv_filt)

	def loss(self, pred, true_label=None, sub_samp=None):
		label_pos	= true_label[0]; 
		label_neg	= true_label[1:]
		loss 		= self.bceloss(pred, true_label)
		return loss

	def circular_padding_chw(self, batch, padding):
		upper_pad	= batch[..., -padding:, :]
		lower_pad	= batch[..., :padding, :]
		temp		= torch.cat([upper_pad, batch, lower_pad], dim=2)

		left_pad	= temp[..., -padding:]
		right_pad	= temp[..., :padding]
		padded		= torch.cat([left_pad, temp, right_pad], dim=3)
		return padded

	def forward(self, sub, rel, neg_ents, strategy='one_to_x'):
		sub_emb		= self.ent_embed(sub)
		rel_emb		= self.rel_embed(rel)
		comb_emb	= torch.cat([sub_emb, rel_emb], dim=1)
		chequer_perm	= comb_emb[:, self.chequer_perm]
		stack_inp	= chequer_perm.reshape((-1, self.p.perm, 2*self.p.k_w, self.p.k_h))
		stack_inp	= self.bn0(stack_inp)
		x		= self.inp_drop(stack_inp)
		x		= self.circular_padding_chw(x, self.p.ker_sz//2)
		x		= F.conv2d(x, self.conv_filt.repeat(self.p.perm, 1, 1, 1), padding=self.padding, groups=self.p.perm)
		x		= self.bn1(x)
		x		= F.relu(x)
		x		= self.feature_map_drop(x)
		x		= x.view(-1, self.flat_sz)
		x		= self.fc(x)
		x		= self.hidden_drop(x)
		x		= self.bn2(x)
		x		= F.relu(x)

		if strategy == 'one_to_n':
			x = torch.mm(x, self.ent_embed.weight.transpose(1,0))
			x += self.bias.expand_as(x)
		else:
			x = torch.mul(x.unsqueeze(1), self.ent_embed(neg_ents)).sum(dim=-1)
			x += self.bias[neg_ents]

		pred	= torch.sigmoid(x)

		return pred