from http import HTTPStatus
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
import numpy as np
import re

Prompts = ['接下来的执行流程如下:\
            1、我输入A_i(i 是从0-200的整数,具体是多少取决于这是第几个输入的数据)。\
            2、请整合[A_0,A_1,A_2...A_{i-1},A_i](请自行确定整合方法)后给出b_i,c_i(两个大于0小于1且保留到小数点后四位的浮点数);要求:每次生成b_i,c_i都是不同的且b_i + c_i = 1 。\
            3、我会给你打分(0-1的浮点数)。\
            4、我输入A_{i+1},然后回到2、(2、中的A_i变成A_{i+1}, 请尽可能获取更高的分数)。\
                   补充说明:\
                   A_i的长度均为208(其中i是正整数,表示这是我输入的第几个数据;\
                   A[0:2]均是正整数代表两个损失函数的值,\
                   A[2:5]均是正整数代表第一个损失对应的分类头在每个超类别上的性能,\
                   A[5:8]均是正整数代表第二个损失对应的分类头在每个超类别上的性能,\
                   A[8:108]均是正整数代表第一个损失对应的分类头在每个小类别上的性能,\
                   A[108:]均是正整数代表第二个损失对应的分类头在每个小类别上的性能).\
                   ',# 375 Tokens 

       
            '接下来请以“[a,b]”的格式给我两个大于0小于1且和为1的浮点数(保留到小数点后四位)。然后我会给这两个数字打分(0-1)。然后你需要重新给我不同的“[a,b]”然后我会重新对它打分,请尽可能获取更高的分数'# 56 Tokens
            ]


class LLM_ALIBABA:
    # Prompts_id in [0,1]
    def __init__(self,Prompts_id): 
        self.Prompts_id = Prompts_id
        self.messages = [#{'role': Role.SYSTEM, 'content': '你现在是一个算法工程师，请严格按照用户限制的输入以及输出的格式'}
            ]
        self.call(Prompts[Prompts_id])

        #self.messages = [#{'role': Role.SYSTEM, 'content': '你现在是一个算法工程师，请严格按照用户限制的输入以及输出的格式'}
        #                 {'role': Role.USER, 'content': Prompts[Prompts_id]}]
        self.Prompt_index = 0

    def __call__(self,epoch_info):
        if self.Prompts_id == 0:
            self.call('A_{} = '.format(self.Prompt_index) +  str(list(np.around(epoch_info,0).astype(int))) + '请严格按照“[b_{},c_{}]”的格式回复你生成的那两个数字,然后等待我评分。'.format(self.Prompt_index,self.Prompt_index,self.Prompt_index,self.Prompt_index))

        self.Prompt_index += 1
    
    def evaluete(self,value):
        if self.Prompts_id == 0:
            self.call('此时这两个数字的分数为{:.2f},请等待我输入A_{}。'.format(value,self.Prompt_index+1))
        elif self.Prompts_id == 1:
            self.call('这两个数字的分数为:{:.2f},请重新给我两个数字'.format(value))
    
    def get_weight(self):
        indexs = [-3,-1]
        output = self.messages[indexs[self.Prompts_id]]['content']
        try :
            return eval(output[re.search('\[0\.\d*',output).span()[0]:re.search('0\.\d*\]',output).span()[1]])
        except:
            raise RuntimeError('alibaba llm erro: line 56.  help: please try twice to fix this erro ,{}'.format(output) )

    def call(self,epoch_info):

        self.messages.append({'role': Role.USER, 'content': epoch_info})
        # make second round call
        
        has_response = False
        for _ in range(5):
            response = Generation.call(
                Generation.Models.qwen_max,
                messages=self.messages,
                result_format='message',  # set the result to be "message" format.
                )

            if response.status_code == HTTPStatus.OK:
                has_response = True
                break
            else:
                continue    
        
        if has_response :
            self.messages.append({'role': response.output.choices[0]['message']['role'],
                         'content': response.output.choices[0]['message']['content']})
        else :
            raise RuntimeError('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))


def test(mode = 0):
    llm = LLM_ALIBABA(mode)
    for i in range(100):
        info = (i/200 + np.random.rand(209)*0.5)*100
        llm(info)
        value = i/200 + np.random.rand()*0.5
        llm.evaluete(value)
        print('get_weight:{}'.format(llm.get_weight()))


if __name__ == '__main__':


    test(1)

    