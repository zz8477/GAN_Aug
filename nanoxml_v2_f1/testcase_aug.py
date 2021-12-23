#-*- coding:utf-8 -*-#
# Copyright (c) National University of Defense Technology(NUDT).
# All rights reserved.
#
"""
Created on 2021-11-11

@author: zhangzhuo

usage : 
    python testcase_aug.py dev
    or
    python testcase_aug.py
"""
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import sys
from configparser import ConfigParser
import logging.config
import torch
import numpy as np
# initialize logger
logging.config.fileConfig("logging.cfg")
logger = logging.getLogger("root")

class RuntimeContext(object):
    """ runtime enviroment
    """
    
    def __init__(self):
        """ initialization
        """
        # configuration initialization
        config_parser = ConfigParser()
        config_file = self.get_config_file_name()
        config_parser.read(config_file, encoding="UTF-8")
        sections = config_parser.sections()
        
        coverage_information_matrix_section = sections[0]
        self.covMatrix = config_parser.get(coverage_information_matrix_section, "covMatrix")
        self.INUNITS = config_parser.get(coverage_information_matrix_section, "INUNITS")
        self.TESTNUM_TOTAL = config_parser.get(coverage_information_matrix_section, "TESTNUM_TOTAL")
        
        test_cases_results_section = sections[1]
        self.error = config_parser.get(test_cases_results_section, "error")
        
        resample_test_cases_results_section = sections[2]
        self.covMatrix_new = config_parser.get(resample_test_cases_results_section, "covMatrix_new")
        
        gan_parameter_section = sections[3]
        self.seed_value = config_parser.get(gan_parameter_section, "seed_value")
        self.batch_size = config_parser.get(gan_parameter_section, "batch_size")
        self.num_epoch = config_parser.get(gan_parameter_section, "num_epoch")
        self.z_dimension = config_parser.get(gan_parameter_section, "z_dimension")
        
    def get_config_file_name(self):
        """ get the configuration file name according to the command line parameters
        """
        argv = sys.argv
        config_type = "dev" # default configuration type
        if None != argv and len(argv) > 1 :
            config_type = argv[1]
        config_file = config_type + ".cfg"
        logger.info("get_config_file_name() return : " + config_file)
        return config_file

def main():
    runtime_context = RuntimeContext()
    """ 
    load covMatrix.txt and error.txt
    """
    logger.info("load coverage information matrix")
    f1 = open(runtime_context.covMatrix,'r')
    f2 = open(runtime_context.error,'r')
    f3 = open(runtime_context.covMatrix_new,'w')
    """ 
    set parameters
    """
    logger.info("set parameters")
    seed_value = int(runtime_context.seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    
    batch_size=int(runtime_context.batch_size)
    num_epoch=int(runtime_context.num_epoch)
    z_dimension=int(runtime_context.z_dimension)
    
    
    INUNITS = int(runtime_context.INUNITS)     # coverage data input  change !!!!!!!!!!!!
    TESTNUM_TOTAL = int(runtime_context.TESTNUM_TOTAL)    # test case number  change!!!!!!!!!!!!!
    
    first_ele = True
    for data in f1.readlines():
        data = data.strip('\n')
        nums = data.split()
        if first_ele:
            nums = [float(x) for x in nums]
            matrix_x = np.array(nums)
            first_ele = False
        else:
            nums = [float(x) for x in nums]
            matrix_x = np.c_[matrix_x,nums]
    f1.close()
    
    first_ele = True
    for data in f2.readlines():
        data = data.strip('\n')
        nums = data.split()
        if first_ele:
            nums = [float(x) for x in nums]
            matrix_y = np.array(nums)
            first_ele = False
        else:
            nums = [float(x) for x in nums]
            matrix_y = np.c_[matrix_y,nums]
    f2.close()
    
    
    matrix_x = matrix_x.transpose()
    matrix_y = matrix_y.transpose()
    
    inputs_pre = []
    testcase_fail_num = 0
    for testcase_num in range(len(matrix_y)):
        if matrix_y[testcase_num][0] == 1:
            inputs_pre.append(matrix_x[testcase_num])
            testcase_fail_num = testcase_fail_num + 1
    TESTNUM = testcase_fail_num
    inputs = torch.FloatTensor(inputs_pre)
    labels = torch.FloatTensor(matrix_y)
    """ 
    build minimum suspicious set
    """
    logger.info("build minimum suspicious set")
    minimum_suspicious_set = Variable(torch.zeros(INUNITS))
    for element_num in range(len(inputs[0])):
        flag = 0
        for item in inputs:
            if item[element_num] == 1:
                flag = flag +1
        if flag == testcase_fail_num:
            minimum_suspicious_set[element_num] = 1
    """ 
    Definition discriminator
    """
    class discriminator(nn.Module):
        def __init__(self):
            super(discriminator,self).__init__()
            self.dis=nn.Sequential(
                nn.Linear(INUNITS,256),#The number of input features is INPUTS and the output is 256
                nn.LeakyReLU(0.2),#Nonlinear mapping
                nn.Linear(256,256),#make a linear mapping
                nn.LeakyReLU(0.2),
                nn.Linear(256,1),
                nn.Sigmoid()#Igmoid can map real numbers to [0,1]
            )
        def forward(self, x):
            x=self.dis(x)
            return x
    """ 
    Definition generator
    """
    class generator(nn.Module):
        def __init__(self):
            super(generator,self).__init__()
            self.gen=nn.Sequential(
                nn.Linear(100,256),#The input is mapped to 256 dimensions by linear transformation
                nn.ReLU(True),#relu activation
                nn.Linear(256,256),#linear transformation 
                nn.ReLU(True),#relu activation
                nn.Linear(256,INUNITS),#linear transformation 
                nn.Tanh()#Tanh activation makes the generated data distributed between [- 1,1]
            )
        def forward(self, x):
            x=self.gen(x)
            return x
    """ 
    create object
    """
    D=discriminator()
    G=generator()
    if torch.cuda.is_available():
        D=D.cuda()
        G=G.cuda()

    """ 
    train discriminator
    """
    logger.info("start training")
    criterion = nn.BCELoss() #Single objective binary classification cross entropy function
    d_optimizer=torch.optim.Adam(D.parameters(),lr=0.0003)
    g_optimizer=torch.optim.Adam(G.parameters(),lr=0.0003)
    ###########################Enter the judgment process of training and discriminating#####################
    for epoch in range(num_epoch): 
            real_testcase = inputs.cuda()
            real_label = Variable(torch.ones(TESTNUM)).cuda()
            fake_label = Variable(torch.zeros(TESTNUM)).cuda()
            real_label_new = []
            for item in real_label:
                temp=[]
                temp.append(item)
                real_label_new.append(temp)
            real_label_new = torch.tensor(real_label_new).cuda()
            real_label = real_label_new
            fake_label_new = []
            for item in fake_label:
                temp=[]
                temp.append(item)
                fake_label_new.append(temp)
            fake_label_new = torch.tensor(fake_label_new).cuda()
            
            real_out = D(real_testcase)  # Put the real test case into the discriminator
            d_loss_real = criterion(real_out, real_label)# Get the loss of the real test case
            real_scores = real_out  # The discrimination value of the real test case is obtained. The closer the output value is to 1, the better
            
            z = Variable(torch.randn(TESTNUM, z_dimension)).cuda()  # Generate some noise randomly
            fake_testcase = G(z)  # The random noise is put into the generation network to generate a false test case
            fake_out = D(fake_testcase)  # The discriminator judges the augmentation test case
            d_loss_fake = criterion(fake_out, fake_label_new)  # Get the loss of augmentation test cases
            fake_scores = fake_out  # Get the discrimination value of the false test case. For the discriminator, the closer the loss of the false test case is to 0, the better
            """ 
            loss function and optimization
            """ 
            d_loss = d_loss_real + d_loss_fake #The loss includes the loss of judging true and the loss of judging false
            d_optimizer.zero_grad()  # Before back propagation, return the gradient to 0
            d_loss.backward()  # Back propagation of error
            d_optimizer.step()  # Update parameters
    
            """ 
            train generator
            """
            z = Variable(torch.randn(TESTNUM, z_dimension)).cuda()  # Generate some noise randomly
            fake_testcase = G(z) 
            output = D(fake_testcase)  
            g_loss = criterion(output, real_label)  
            """
            bp and optimize
            """
            g_optimizer.zero_grad()  
            g_loss.backward()  
            g_optimizer.step()  
            logger.info('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                      'D real: {:.6f},D fake: {:.6f}'.format(
                    epoch,num_epoch,d_loss.data.item(),g_loss.data.item(),
                    real_scores.data.mean(),fake_scores.data.mean()  #What is printed is the average loss of the real test case
                ))
    """ 
    start generating
    """
    logger.info("start generating")
    z = Variable(torch.randn((TESTNUM_TOTAL-TESTNUM*2), z_dimension)).cuda()
    fake_testcase = G(z) #Random noise is input into the generator to get a fake test case
    
    fake_testcase = fake_testcase.cpu().detach()
    fake_testcase_numpy = fake_testcase.numpy()
    
    for item in fake_testcase_numpy:
        for element_num in range(len(item)):
            if item[element_num] <= 0.5:
                f3.write('0')
            else:
    #            f3.write('1')
                f3.write(str(round(item[element_num],2)))
            f3.write(' ')
        f3.write('\n')
    logger.info(fake_testcase_numpy)
    f3.close()
    """
    save model
    """
    torch.save(G.state_dict(),'./generator.pth')
    torch.save(D.state_dict(),'./discriminator.pth')
    logger.info(" generate complete")
if __name__ == "__main__":
    main()
