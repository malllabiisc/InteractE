from helper import *

from collections import defaultdict as ddict
from torch.utils.data import Dataset

class TrainDataset(Dataset):
	"""
	Training Dataset class.

	Parameters
	----------
	triples:	The triples used for training the model
	params:		Parameters for the experiments
	
	Returns
	-------
	A training Dataset class instance used by DataLoader
	"""
	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params
		self.strategy	= self.p.train_strategy
		self.entities	= np.arange(self.p.num_ent, dtype=np.int32)

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele			= self.triples[idx]
		triple, label, sub_samp	= torch.LongTensor(ele['triple']), np.int32(ele['label']), np.float32(ele['sub_samp'])
		trp_label		= self.get_label(label)

		if self.p.lbl_smooth != 0.0:
			trp_label = (1.0 - self.p.lbl_smooth)*trp_label + (1.0/self.p.num_ent)

		if self.strategy == 'one_to_n':
			return triple, trp_label, None, None

		elif self.strategy == 'one_to_x':
			sub_samp		= torch.FloatTensor([sub_samp])
			neg_ent			= torch.LongTensor(self.get_neg_ent(triple, label))
			return triple, trp_label, neg_ent, sub_samp
		else: 
			raise NotImplementedError


	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		trp_label	= torch.stack([_[1] 	for _ in data], dim=0)

		if not data[0][2] is None:							# one_to_x
			neg_ent		= torch.stack([_[2] 	for _ in data], dim=0)
			sub_samp	= torch.cat([_[3] 	for _ in data], dim=0)
			return triple, trp_label, neg_ent, sub_samp
		else:
			return triple, trp_label
	
	def get_neg_ent(self, triple, label):
		def get(triple, label):
			if self.strategy == 'one_to_x':
				pos_obj		= triple[2]
				mask		= np.ones([self.p.num_ent], dtype=np.bool)
				mask[label]	= 0
				neg_ent		= np.int32(np.random.choice(self.entities[mask], self.p.neg_num, replace=False)).reshape([-1])
				neg_ent		= np.concatenate((pos_obj.reshape([-1]), neg_ent))
			else:
				pos_obj		= label
				mask		= np.ones([self.p.num_ent], dtype=np.bool)
				mask[label]	= 0
				neg_ent		= np.int32(np.random.choice(self.entities[mask], self.p.neg_num - len(label), replace=False)).reshape([-1])
				neg_ent		= np.concatenate((pos_obj.reshape([-1]), neg_ent))

				if len(neg_ent) > self.p.neg_num:
					import pdb; pdb.set_trace()
					
			return neg_ent

		neg_ent = get(triple, label)
		return neg_ent

	def get_label(self, label):
		if self.strategy == 'one_to_n':
			y = np.zeros([self.p.num_ent], dtype=np.float32)
			for e2 in label: y[e2] = 1.0
		elif self.strategy == 'one_to_x':
			y = [1] + [0] * self.p.neg_num
		else: 
			raise NotImplementedError
		return torch.FloatTensor(y)


class TestDataset(Dataset):
	"""
	Evaluation Dataset class.

	Parameters
	----------
	triples:	The triples used for evaluating the model
	params:		Parameters for the experiments
	
	Returns
	-------
	An evaluation Dataset class instance used by DataLoader for model evaluation
	"""
	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele		= self.triples[idx]
		triple, label	= torch.LongTensor(ele['triple']), np.int32(ele['label'])
		label		= self.get_label(label)

		return triple, label

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		label		= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, label
	
	def get_label(self, label):
		y = np.zeros([self.p.num_ent], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)
