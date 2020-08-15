import os
import sys
import kaldi_io
import numpy as np
import random
import scipy.spatial
import math


epsilon = 20

look_up = dict()
vec_cen = {k:v for k,v in kaldi_io.read_vec_flt_ark("x_cen.ark")}

def get_vec(file_name):
	vec = {k:v for k,v in kaldi_io.read_vec_flt_ark(file_name)}
	return vec

def v_transfer(epsilon, utt, value):
	prob = dict()
	all_prob = 0
	key = ("-").join(utt.split("-")[0:1])
	if key in look_up.keys():
		return look_up[key]

	for k, v in vec_cen.items():
		temp = np.arccos(1 - scipy.spatial.distance.cosine(value, v)) / np.pi
		if math.isinf(temp):
			print("Error")

		prob[k] = math.exp((epsilon) * (1-temp))
		all_prob = all_prob + prob[k]
	temp = random.uniform(0, all_prob)
	cumprob = 0.0
	ans = value
	for k, p in prob.items():
		cumprob += p
		ans = vec_cen[k]
		if temp < cumprob:
			look_up[key] = ans
			vec_cen[k] = value
			break
	return(ans)

def store(file_name, vec_trans):
	ark_file = file_name
	with kaldi_io.open_or_fd(ark_file, 'wb') as f:
		for k, v in vec_trans.items():
			kaldi_io.write_vec_flt(f, v, str(k))
	with kaldi_io.open_or_fd("x_cen.ark", 'wb') as f:
		for k, v in vec_cen.items():
			kaldi_io.write_vec_flt(f, v, str(k))

file_name = "exp/xvector_nnet_1a/xvectors_sample/xvector.1.ark"
vec = get_vec(file_name)
vec_trans = dict()
for k, v in vec.items():
	vec_trans[k] = v_transfer(epsilon, k, v)
store(file_name, vec_trans)

