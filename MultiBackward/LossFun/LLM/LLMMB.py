from .alibaba import *
import os
import dashscope

# alibaba
dashscope.api_key = 'sk-b2032ca288e24f67ab16dab6b36f7a76'

# openai 
os.environ['OPENAI_API_KEY'] = 'sk-b2032ca288e24f67ab16dab6b36f7a76'




class LMB:
    def __init__(self,llm,mode):
        self.weights = []
        if llm == 'alibaba':
            self.llm = LLM_ALIBABA(mode)
            if mode == 0:
                self.weights.append([0.5,0.5])
            elif mode == 1:
                self.weights.append(self.llm.get_weight())

    
    def __call__(self,info,value):
        self.llm(info)    
        self.llm.evaluete(value)
        weight = self.llm.get_weight()
        self.weights.append(weight)
        return weight
    
    def backward(self,losses):
        for index,w in enumerate(self.weights[-1]):
            losses[index] *= w 
        return losses


if __name__ == '__main__':
    lmb = LMB('alibaba',0)
    for i in range(100):
        info = (i/200 + np.random.rand(209)*0.5)*100
        value = i/200 + np.random.rand()*0.5
        print('get_weight:{}'.format(lmb(info,value)))
    
    