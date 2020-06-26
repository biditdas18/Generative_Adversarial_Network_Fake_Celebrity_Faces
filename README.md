"# Generative_Adversarial_Network_Fake_Celebrity_Faces"

GAN 


Generative Adversarial Network is one of the most interesting ideas in the last 10 years as stated by Yann LeCun, V.P & chief AI scientist at Facebook. It is one of the unconventional networks as instead of analyzing data and making predictions it’s one of those networks that generates data. GANs basically have the capability to create images that look like the photograph of human faces. Interestingly the faces generated do not belong to any real person. Though seemingly complex GANs are built on the basic concepts of the neural network.


Theory 


Generating new data is basically a game of probability and generating faces can be expresses as a random variable generation problem. In GANs we basically take an N dimensional vector space and generate a new celebrity face by generating a new vector following the celebrity face probability distribution. This process is kind of a mapping done from N dimensional vector space to the celebrity faces that we have in our dataset. This step is exactly the opposite step to that of the feature embedding process used in face recognition where we map a face to a N dimensional feature vector. 


Architecture and Working


A Generative Adversarial Network basically consists of two components which are basically two Neural Networks namely Generator and Discriminator. The function of the generator is to map the N dimensional feature vector to celebrity faces in our dataset and generate faces based on the probability distribution of the of the images in our dataset. Similarly, the function of the discriminator is to discriminate between the generated faces and the real faces in the dataset
Since we have two neural networks in action in GANs we have two loss functions, where one loss functions belong to that of the discriminator the other loss function belong to the generator. While for the discriminator our objective is to minimize the classification error, for the generator our objective is to maximize the classification error i.e. we want the fake faces generated to be considered as real. This is where the adversarial factor kicks in. We have two neural networks working in conjugation where the objective of one is to minimize the classification error the objective of the other is to maximize the error. An equilibrium is reached between both the networks when the generator produces the sample that follows the celebrity face probability distribution and the discriminator is able to predict real or face faces with equal probability. Moreover, it should be noted that both the networks learn equally during the training and converge together. In other words, both the network help improve the other in each step hence unlike a conventional neural network we don’t see the loss function going down.
Once the network is trained for sufficient number of epochs, we can see our generator generating faces closer to that of human faces.
The last thing that needs to be highlighted in this project is the procedure in which the training process in carried out. We first train the discriminator independently using binary cross entropy loss function. Once the discriminator is trained, we freeze the weights of the discriminator and train the combined model of the generator and the discriminator using the same loss function but this time keeping our target as 1 which is the classification value for real images. Other details of the code are commented in the notebook



