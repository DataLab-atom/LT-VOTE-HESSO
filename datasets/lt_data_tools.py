import numpy as np

def get_supper_targets(targets,m,num_class):
    
    targets = np.array(targets)
    '''
    fun to get a supper_targets on targets
    
    get  c_list = np.array([(targets == i).sum() for i in range(100)])

    use func  get  c_list_2 = np.array([(targets_2 == i).sum() for i in range(100)])

    to set min(\sum(\std(c_list_2) ,dim = 0)) and  min(\std(\sum(c_list_2,dim = 1))) 

    '''

    c_list = np.array([(targets == i).sum() for i in range(num_class)])
    
    indexs =  np.linspace(0,num_class,m + 1,dtype=np.int16) 
    sum_c_i =  np.zeros(m)

    for i in range(m):
        sum_c_i[i] = sum(c_list[indexs[i]:indexs[i + 1]])

    std_c = sum_c_i.std()

    changes = np.ones(m + 1)
    while sum(changes) > 0:
        changes = np.zeros(m + 1)

        for i in range(1,m):   
            temp = c_list[indexs[i]]
            
            if (indexs[i]  - 1) > indexs[i - 1]:
                #尝试左移
                indexs[i] -= 1           
                sum_c_i[i - 1] -= temp
                sum_c_i[i] += temp
                if std_c > sum_c_i.std():
                    std_c = sum_c_i.std()
                    changes[i] = 1
                    continue
                
                sum_c_i[i - 1] += temp
                sum_c_i[i] -= temp
                indexs[i] += 1

            if (indexs[i] + 1) < indexs[i + 1]:
                #尝试右移动
                indexs[i] += 1
                sum_c_i[i - 1] += temp
                sum_c_i[i] -= temp
                
                if std_c > sum_c_i.std():
                    std_c = sum_c_i.std()
                    changes[i] = 1
                    continue
                
                sum_c_i[i - 1] -= temp
                sum_c_i[i] += temp
                indexs[i] -= 1


    targets_2_table = np.zeros_like(c_list)
    for i in range(m):
        targets_2_table[indexs[i]:indexs[i + 1]] = i

    targets_2 = targets_2_table[targets]
    return targets_2


def get_supper_targets_2_table(targets,m,num_class):

    targets = np.array(targets)
    '''
    fun to get a supper_targets on targets
    
    get  c_list = np.array([(targets == i).sum() for i in range(100)])

    use func  get  c_list_2 = np.array([(targets_2 == i).sum() for i in range(100)])

    to set min(\sum(\std(c_list_2) ,dim = 0)) and  min(\std(\sum(c_list_2,dim = 1))) 

    '''

    c_list = np.array([(targets == i).sum() for i in range(num_class)])

    
    num_class = c_list.shape[0]
    indexs =  np.linspace(0,num_class,m + 1,dtype=np.int16) 
    sum_c_i =  np.zeros(m)

    for i in range(m):
        sum_c_i[i] = sum(c_list[indexs[i]:indexs[i + 1]])

    std_c = sum_c_i.std()

    changes = np.ones(m + 1)
    while sum(changes) > 0:
        changes = np.zeros(m + 1)

        for i in range(1,m):   
            temp = c_list[indexs[i]]
            
            if (indexs[i]  - 1) > indexs[i - 1]:
                #尝试左移
                indexs[i] -= 1           
                sum_c_i[i - 1] -= temp
                sum_c_i[i] += temp
                if std_c > sum_c_i.std():
                    std_c = sum_c_i.std()
                    changes[i] = 1
                    continue
                
                sum_c_i[i - 1] += temp
                sum_c_i[i] -= temp
                indexs[i] += 1

            if (indexs[i] + 1) < indexs[i + 1]:
                #尝试右移动
                indexs[i] += 1
                sum_c_i[i - 1] += temp
                sum_c_i[i] -= temp
                
                if std_c > sum_c_i.std():
                    std_c = sum_c_i.std()
                    changes[i] = 1
                    continue
                
                sum_c_i[i - 1] -= temp
                sum_c_i[i] += temp
                indexs[i] -= 1


    targets_2_table = np.zeros_like(c_list)
    for i in range(m):
        targets_2_table[indexs[i]:indexs[i + 1]] = i
        
    return targets_2_table